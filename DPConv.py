import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class DPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, extension, stride):
        """
        类卷积设计注意力模块。
        Args:
            in_channels: 输入张量输入通道数
            out_channels: 输出通道数，主要用于设定多头注意力输出通道数
            kernel: 输入进入多头注意力机制的张量的尺寸大小
            stride: 选取步长
            extension: 扩充大小，获取更大范围的特征信息
        """
        super(DPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel
        self.stride = stride
        self.extension = extension

        # self.attention = Multihead_Attention(in_channels, out_channels)

    def forward(self, x):
        # 获取输入尺寸
        N, C, H, W = x.shape

        # 对边界进行填充，这一步是为了防止extension获取更大范围特征时边界取值异常
        x = F.pad(x, (self.extension, self.extension, self.extension, self.extension), mode='replicate')

        # 使用 unfold 提取所有滑动窗口区域
        unfolded = F.unfold(x, kernel_size=self.kernel_size + 2 * self.extension, stride=self.stride)
        num_windows = unfolded.shape[2]  # 滑动窗口的数量

        # 调整形状，使每个窗口的区域适合池化处理
        unfolded = unfolded.view(N, C, self.kernel_size + 2 * self.extension, self.kernel_size + 2 * self.extension,
                                 num_windows)

        # 对每个窗口进行平均池化，池化到 (kernel_size, kernel_size) 大小
        pooled_regions = F.adaptive_avg_pool2d(
            unfolded.permute(0, 4, 1, 2, 3).contiguous().view(N, -1, self.kernel_size + 2 * self.extension,
                                                              self.kernel_size + 2 * self.extension),
            output_size=(self.kernel_size, self.kernel_size))

        # 将 attention_output 调整为 (N, C * kernel_size**2, num_windows)
        # attention_output = self.attention(pooled_regions)
        attention_output = pooled_regions
        attention_output = attention_output.view(N, C * self.kernel_size * self.kernel_size, num_windows)

        # 重新通过 fold 将滑动窗口展开为完整的张量
        output = F.fold(attention_output, output_size=(H, W),
                        kernel_size=self.kernel_size, stride=self.stride)
        # print("output shape: " + str(output.shape))

        # 计算重叠部分的权重
        count = F.fold(torch.ones_like(attention_output), output_size=(H, W),
                       kernel_size=self.kernel_size, stride=self.stride)
        # print("count shape: " + str(count.shape))

        # 对重叠部分取平均值
        output = output / count

        return output


class Multihead_Attention(nn.Module):
    """
    对输入张量进行多头自注意力
    """

    def __init__(self, in_channels, out_channels):
        super(Multihead_Attention, self).__init__()

        self.dim = self.num_head = in_channels // 64
        self.h = self.dim * self.num_head
        self.scale = self.dim ** -0.5
        self.qkv = nn.Conv2d(in_channels, 3 * self.h, kernel_size=1, stride=1)
        self.resume = nn.Conv2d(self.h, out_channels, kernel_size=1, stride=1)
        self.position = nn.Conv2d(self.h, self.h, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 输入 x 的形状为 (N, num_patches, C, H, W)
        N, C, H, W = x.shape

        # 将输入 reshape，合并 batch 维度和 num_patches 维度
        x = x.reshape(N, C, H, W)

        # 计算 Q、K、V
        qkv = self.qkv(x)
        qkv = qkv.view(N, self.num_head, -1, H * W)  # (N*num_patches, num_head, 3*dim, H*W)
        q, k, v = qkv.split([self.dim, self.dim, self.dim], dim=2)  # 分割成 Q、K、V

        # 计算注意力分数
        attn = (q.transpose(-2, -1) @ k) * self.scale  # (N*num_patches, H*W, H*W)
        attn = attn.softmax(dim=-1)

        # 计算注意力输出
        out = (v @ attn.transpose(-2, -1)).view(N, self.h, H, W)
        out = out + self.position(v.reshape(N, self.h, H, W))

        # 恢复原始形状
        out = self.resume(out)

        return out


# 测试代码
if __name__ == '__main__':
    x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    # x = torch.randn(1, 576, 8, 8)

    attention = DPConv(in_channels=x.shape[1], out_channels=x.shape[1], kernel=2, stride=2, extension=1)
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
    x = attention(x)
    print('output shape:' + str(x.shape))
