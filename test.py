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
        print(f"shape: {y.shape}")
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
    def __init__(self, in_channels, out_channels, kernel, stride):
        """
        类卷积设计注意力模块。
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_windows: unfold展开窗口数。H, W方向上各产生num_windows个窗口
            extension: 扩充大小，获取更大范围的特征信息
        """
        super(DPConv, self).__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = (self.in_channels // 4) * 4
        self.out_channels = out_channels
        self.kernel = kernel
        self.kernel_list = [self.kernel // 2, self.kernel, self.kernel + self.kernel // 2]
        self.stride = stride

        self.reduce = nn.Conv2d(self.in_channels, self.in_channels // 4, kernel_size=1, stride=1)
        self.pe = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1,
                            groups=self.in_channels)
        # self.attention = Attention(self.in_channels, num_heads=self.out_channels // 64)

    def forward(self, x):
        N, C, H, W = x.shape
        # 尺寸修剪，确保均为偶数
        if H % 2 != 0:
            x = x[:, :, :-1, :]
        if W % 2 != 0:
            x = x[:, :, :, :-1]
        # print("修剪后的x的形状：" + str(x.shape))
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.reduce(x)
        N, C, H, W = x.shape
        print("填充后的x的形状：" + str(x.shape))

        # 多尺度unfold核展开
        x_unfolded = []
        for i in range(len(self.kernel_list)):
            print("第" + str(i) + "次展开：")
            # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
            # (N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)
            unfolded = F.unfold(x, kernel_size=self.kernel_list[i], stride=self.stride)
            print("unfolded的形状: " + str(unfolded.shape))
            num_windows = unfolded.shape[2]
            # view为(N, C, self.kernel_list[i], self.kernel_list[i], num_windows),便于后续处理
            unfolded = unfolded.view(N * num_windows, C, self.kernel_list[i], self.kernel_list[i])
            print("view之后的unfolded的形状: " + str(unfolded.shape))

            if self.kernel_list[i] < self.kernel:
                unfolded = F.interpolate(unfolded, size=(self.kernel, self.kernel), mode='bilinear').view(N, num_windows, C, self.kernel, self.kernel)
                print("上采样之后的unfold的形状：" + str(unfolded.shape))
                x_unfolded.append(unfolded)
            elif self.kernel_list[i] > self.kernel:
                unfolded = F.adaptive_avg_pool2d(unfolded, output_size=(self.kernel, self.kernel)).view(N, num_windows, C, self.kernel, self.kernel)
                x_unfolded.append(unfolded)
                print("池化之后的unfold的形状：" + str(unfolded.shape))
            else:
                x_unfolded.append(unfolded.view(N, num_windows, C, self.kernel, self.kernel))
                print("unfold的形状：" + str(unfolded.view(N, num_windows, C, self.kernel, self.kernel).shape))
        x_unfolded_stack = torch.cat(x_unfolded, dim=1)
        print("堆叠后的x_unfolded_stack的形状：" + str(x_unfolded_stack.shape))
        pass


# 测试代码
if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(1, 576, 9, 8)

    attention = DPConv(x.shape[1], x.shape[1], 4, 2)
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
