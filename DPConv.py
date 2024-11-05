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
    def __init__(self, in_channels, out_channels, kernel_list, stride, extension):
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
        self.out_channels = out_channels
        self.kernel_list = kernel_list
        self.stride = stride
        self.extension = extension

        self.attention = Attention(self.in_channels, num_heads=self.out_channels // 64)
        self.pe = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1,
                            groups=self.in_channels)

    def forward(self, x):
        output = [x]
        N_orin, C_orin, H_orin, W_orin = x.shape
        # 对边界进行填充，以获取更大范围的特征信息
        x = F.pad(x, (self.extension, self.extension, self.extension, self.extension), mode='replicate')
        N_pad, C_pad, H_pad, W_pad = x.shape

        for i in range(len(self.kernel_list)):
            # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
            # (N, C * (self.kernel_list[i] + 2 * self.extension) * (self.kernel_list[i] + 2 * self.extension)), num_windows)
            unfolded = F.unfold(x, kernel_size=self.kernel_list[i] + 2 * self.extension, stride=self.stride)
            num_windows = unfolded.shape[2]
            # view为(N, C, self.kernel_list[i] + 2 * self.extension, self.kernel_list[i] + 2 * self.extension, num_windows),便于后续处理
            unfolded = unfolded.view(N_pad, C_pad, self.kernel_list[i] + 2 * self.extension,
                                     self.kernel_list[i] + 2 * self.extension, num_windows)
            # print("unfolded的形状: " + str(unfolded.shape))

            # 对每个窗口进行平均池化，池化到 (self.kernel_list[i], self.kernel_list[i]) 大小
            pooled_regions = F.adaptive_avg_pool2d(
                unfolded.permute(0, 4, 1, 2, 3).contiguous().view(N_pad, -1, self.kernel_list[i] + 2 * self.extension,
                                                                  self.kernel_list[i] + 2 * self.extension),
                output_size=(self.kernel_list[i], self.kernel_list[i]))
            # print("pooled_region的形状： " + str(pooled_regions.shape))
            # view为(N, num_windows, C, self.kernel_list[i] * Wself.kernel_list[i]), 便于后续处理
            pooled_regions = pooled_regions.view(-1, C_pad, self.kernel_list[i], self.kernel_list[i])
            # print("view之后的pooled_region的形状： " + str(pooled_regions.shape))
            # print("num_windows数目： " + str(self.num_windows))
            # print("conv2后的pooled_region的形状： " + str(self.conv2(pooled_regions).shape))

            # 添加位置编码
            # pooled_regions = self.conv2(pooled_regions) + self.position(pooled_regions)
            # print("位置编码之后的pooled_region的形状： " + str(pooled_regions.shape))
            # print("pooled_region的形状： " + str(pooled_regions.shape))

            # 输入注意力模块
            attention_output = self.attention((self.pe(pooled_regions) + pooled_regions))
            # attention_output = pooled_regions
            # print("attention的形状： " + str(attention_output.shape))

            # 转化为(N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)，便于进行fold操作
            attention_output = attention_output.view(N_orin, C_orin * self.kernel_list[i] * self.kernel_list[i],
                                                     num_windows)
            # print("view之后的attention的形状： " + str(attention_output.shape))

            # 重新通过 fold 将滑动窗口展开为完整的张量，形状为(N, C, H_orin, W_orin)
            output_tmp = F.fold(attention_output, output_size=(H_orin, W_orin), kernel_size=self.kernel_list[i],
                                stride=self.stride)
            # print("fold后的形状: " + str(output.shape))

            # 计算重叠部分的权重
            count = F.fold(torch.ones_like(attention_output), output_size=(H_orin, W_orin),
                           kernel_size=self.kernel_list[i], stride=self.stride)
            # print("count 形状: " + str(count.shape))

            # 对重叠部分取平均值
            output_tmp = output_tmp / count
            # print("第" + str(i) + "个输出形状：" + str(output_tmp.shape))
            output.append(output_tmp)

        sum_tensor = torch.stack(output).sum(dim=0)
        # print(sum_tensor.shape)
        out = sum_tensor / len(output)
        # print("out 形状: " + str(out.shape))
        return out


# 测试代码
if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(1, 576, 8, 8)

    attention = DPConv(x.shape[1], x.shape[1], [2, 4, 6], 2, 1)
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
