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


class DPConv(nn.Module):
    def __init__(self, in_channels, out_channels, extension, num_windows):
        """
        类卷积设计注意力模块。
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_windows: unfold展开窗口数。H, W方向上各产生num_windows个窗口
            extension: 扩充大小，获取更大范围的特征信息，为平方数
        """
        super(DPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_windows = num_windows
        self.extension = extension

        self.conv1 = nn.Conv2d(self.in_channels, 2 * self.in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(2 * self.in_channels, self.in_channels, kernel_size=1)
        self.position = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.attention = SELayer(self.in_channels, int(self.in_channels ** 0.5))

    def forward(self, x):
        x1, x2 = self.conv1(x).split((self.in_channels, self.in_channels), dim=1)
        print("分裂之后的 x1, x2的形状：" + str(x1.shape) + str(x2.shape))
        N_orin, C_orin, H_orin, W_orin = x1.shape
        # 对边界进行填充，以获取更大范围的特征信息
        x1 = F.pad(x1, (self.extension, self.extension, self.extension, self.extension), mode='replicate')
        N_pad, C_pad, H_pad, W_pad = x1.shape

        # 还原为原尺寸时的fold的kernel和stride尺寸
        H_orin_stride = H_orin // self.num_windows
        H_orin_kernel = H_orin_stride + H_orin % self.num_windows
        W_orin_stride = W_orin // self.num_windows
        W_orin_kernel = W_orin_stride + W_orin % self.num_windows
        print("H_orin: kernel: " + str(H_orin_kernel) + "; stride: " + str(H_orin_stride))
        print("W_orin: kernel: " + str(W_orin_kernel) + "; stride: " + str(W_orin_stride))

        # 填充后的张量unfold时的kernel和stride尺寸
        H_pad_stride = H_pad // self.num_windows
        H_pad_kernel = H_pad_stride + H_pad % self.num_windows
        W_pad_stride = W_pad // self.num_windows
        W_pad_kernel = W_pad_stride + W_pad % self.num_windows
        print("H_pad: kernel: " + str(H_pad_kernel) + "; stride: " + str(H_pad_stride))
        print("W_pad: kernel: " + str(W_pad_kernel) + "; stride: " + str(W_pad_stride))

        # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
        # (N, C*H_pad_kernel*W_pad_kernel, num_windows * num_windows)
        unfolded = F.unfold(x1, kernel_size=(H_pad_kernel, W_pad_kernel), stride=(H_pad_stride, W_pad_stride))
        # view为(N, C, H_pad_kernel, W_pad_kernel, num_windows * num_windows),便于后续处理
        unfolded = unfolded.view(N_pad, C_pad, H_pad_kernel, W_pad_kernel, self.num_windows ** 2)
        print("unfolded的形状: " + str(unfolded.shape))

        # 对每个窗口进行平均池化，池化到 (H_orin_kernel, W_orin_kernel) 大小
        pooled_regions = F.adaptive_avg_pool2d(
            unfolded.permute(0, 4, 1, 2, 3).contiguous().view(N_pad, -1, H_pad_kernel, W_pad_kernel),
            output_size=(H_orin_kernel, W_orin_kernel))
        print("pooled_region的形状： " + str(pooled_regions.shape))
        # view为(N, num_windows * num_windows, C, H_orin_kernel * W_orin_kernel), 便于后续处理
        pooled_regions = pooled_regions.view(-1, C_pad, H_orin_kernel, W_orin_kernel)
        print("view之后的pooled_region的形状： " + str(pooled_regions.shape))
        print("num_windows数目： " + str(self.num_windows))
        print("conv2后的pooled_region的形状： " + str(self.conv2(pooled_regions).shape))

        # 添加位置编码
        pooled_regions = self.conv2(pooled_regions) + self.position(pooled_regions)
        print("位置编码之后的pooled_region的形状： " + str(pooled_regions.shape))
        print("pooled_region的形状： " + str(pooled_regions.shape))

        # 输入注意力模块
        attention_output = self.attention(pooled_regions) + pooled_regions
        print("attention的形状： " + str(attention_output.shape))

        # 转化为(N, C * H_orin_kernel * W_orin_kernel, num_windows * num_windows)，便于进行fold操作
        attention_output = attention_output.view(N_orin, C_orin * H_orin_kernel * W_orin_kernel, self.num_windows ** 2)
        print("view之后的attention的形状： " + str(attention_output.shape))

        # 重新通过 fold 将滑动窗口展开为完整的张量，形状为(N, C, H_orin, W_orin)
        output = F.fold(attention_output, output_size=(H_orin, W_orin),
                        kernel_size=(H_orin_kernel, W_orin_kernel), stride=(H_orin_stride, W_orin_stride))
        print("fold后的形状: " + str(output.shape))

        # 计算重叠部分的权重
        count = F.fold(torch.ones_like(attention_output), output_size=(H_orin, W_orin),
                       kernel_size=(H_orin_kernel, W_orin_kernel), stride=(H_orin_stride, W_orin_stride))
        print("count 形状: " + str(count.shape))

        # 对重叠部分取平均值
        output = output / count

        out = self.conv3(torch.cat((output, x2), 1))
        print("output 形状：" + str(out.shape))

        return out


# 测试代码
if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(1, 576, 20, 20)

    attention = DPConv(x.shape[1], x.shape[1], 1, 6)
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
