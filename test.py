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
        """
        super(DPConv, self).__init__()
        if not (kernel % 4 == 0):
            raise ValueError("Kernel size must be a multiple of 4.")
        self.in_channels = in_channels
        self.bottleneck_channels = (self.in_channels // 4) * 4
        self.out_channels = out_channels
        self.kernel = kernel
        self.kernel_list = [self.kernel // 2, self.kernel, self.kernel + self.kernel // 2]
        self.stride = stride
        self.padding = self.kernel // 2

        self.attention = Attention(self.in_channels, num_heads=self.out_channels // 64)

    def forward(self, x):
        # print("输入特征均值mean:", x.mean().item(), "输入特征方差std:", x.std().item())
        N, C, H, W = x.shape

        # 多尺度unfold核展开
        out_list = [x]
        for i in range(len(self.kernel_list)):

            # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
            # (N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)
            unfolded = F.unfold(x, kernel_size=self.kernel_list[i], stride=self.stride)
            num_windows = unfolded.shape[2]
            # view为(N, C, self.kernel_list[i], self.kernel_list[i], num_windows),便于后续处理
            unfolded = unfolded.view(N * num_windows, C, self.kernel_list[i], self.kernel_list[i])

            # 输入注意力模块
            attention_output = self.attention(unfolded)

            # 转化为(N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)，便于进行fold操作
            attention_output = attention_output.view(N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)

            # 计算重叠部分的权重
            count = F.fold(torch.ones_like(attention_output), output_size=(H, W), kernel_size=self.kernel_list[i], stride=self.stride)
            # print("count 形状: " + str(count.shape))

            # 重新通过 fold 将滑动窗口展开为完整的张量，形状为(N, C, H_orin, W_orin)
            attention_output = F.fold(attention_output, output_size=(H, W), kernel_size=self.kernel_list[i], stride=self.stride)
            # print("fold后的形状: " + str(output.shape))

            # 对重叠部分取平均值
            attention_output = attention_output / count
            out_list.append(attention_output)

        out = sum(out_list) / len(out_list)
        print(out_list[0] - out)

        return out


# 测试代码
if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(4, 576, 20, 20)

    attention = DPConv(x.shape[1], x.shape[1], 4, 2)
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
