import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = nn.Conv2d(dim, h, 1, )
        self.proj = nn.Conv2d(dim, dim, 1, )
        self.pe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
                (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class DPConv(nn.Module):
    def __init__(self, in_channels, num_windows):
        """
        类卷积设计注意力模块。
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(DPConv, self).__init__()
        # if not (num_windows % 2 == 0):
        #     raise ValueError("Kernel size must be even.")
        self.in_channels = in_channels
        self.num_windows_list = [num_windows // 2, num_windows, num_windows + num_windows // 2]
        self.bottlenck_channels = in_channels // len(self.num_windows_list)

        self.conv1 = nn.Conv2d(self.in_channels, self.bottlenck_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(self.bottlenck_channels, self.bottlenck_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(len(self.num_windows_list) * self.bottlenck_channels, self.in_channels, kernel_size=1, stride=1)
        self.position = nn.Conv2d(self.bottlenck_channels, self.bottlenck_channels, kernel_size=3, stride=1, padding=1, groups=self.bottlenck_channels)

        self.attention = SELayer(self.bottlenck_channels, reduction=16)

    def _make_even(self, x):
        """
        确保输入的 H 和 W 为偶数，如果为奇数，则在右边或下方补零。
        """
        N, C, H, W = x.shape
        pad_h = 1 if H % 2 != 0 else 0  # 如果 H 是奇数，补 1
        pad_w = 1 if W % 2 != 0 else 0  # 如果 W 是奇数，补 1

        # 使用 F.pad 进行补偿，右边和下方分别补 0
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return x

    def _get_unfold_config(self, x):
        """
        计算不同尺寸输入下的unfold的核长kernel和步长stride，以便于能处理到每个像素
        """
        N, C, H, W = x.shape
        kernel_list = []
        stride_list = []

        for i in range(len(self.num_windows_list)):
            H_stride = H // self.num_windows_list[i]
            H_kernel = H_stride + H % self.num_windows_list[i]
            W_stride = W // self.num_windows_list[i]
            W_kernel = W_stride + W % self.num_windows_list[i]
            kernel_list.append((H_kernel, W_kernel))
            stride_list.append((H_stride, W_stride))

        return kernel_list, stride_list

    def forward(self, x):
        x_compress = self.conv1(x)
        N, C, H, W = x_compress.shape
        kernel_list, stride_list = self._get_unfold_config(x_compress)
        # print("kernel_list: " + str(kernel_list))
        # print("stride_list: " + str(stride_list))

        # 多尺度unfold核展开
        out_list = []
        for i in range(len(self.num_windows_list)):
            # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
            # (N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)
            unfolded = F.unfold(x_compress, kernel_size=kernel_list[i], stride=stride_list[i])
            # view为(N, C, self.kernel_list[i], self.kernel_list[i], num_windows),便于后续处理
            unfolded = unfolded.view(-1, C, kernel_list[i][0], kernel_list[i][1])

            # 输入注意力模块
            # attention_output = unfolded
            attention_output = self.attention(self.conv2(unfolded)) + self.position(unfolded)

            # 转化为(N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)，便于进行fold操作
            attention_output = attention_output.view(N, C * kernel_list[i][0] * kernel_list[i][1], self.num_windows_list[i] ** 2)

            # 计算重叠部分的权重
            count = F.fold(torch.ones_like(attention_output), output_size=(H, W), kernel_size=kernel_list[i], stride=stride_list[i])
            # print("count 形状: " + str(count.shape))

            # 重新通过 fold 将滑动窗口展开为完整的张量，形状为(N, C, H_orin, W_orin)
            attention_output = F.fold(attention_output, output_size=(H, W), kernel_size=kernel_list[i], stride=stride_list[i])
            # print("fold后的形状: " + str(output.shape))

            # 对重叠部分取平均值
            attention_output = attention_output / count
            out_list.append(attention_output)

        out = self.conv3(torch.cat(out_list, dim=1) + x)
        return out


# 测试代码
if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(1, 576, 8, 8)

    attention = DPConv(x.shape[1], 4)
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
