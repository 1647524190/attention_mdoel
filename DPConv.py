import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from thop import profile
from CBAM import Cbam
from PSA import PSA
from SEAttention import SELayer


class DPConv(nn.Module):
    def __init__(self, c_in, num_windows):
        """
        类卷积设计注意力模块。
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(DPConv, self).__init__()
        self.windows_list = [num_windows // 2, num_windows, num_windows + num_windows // 2]
        bottlenck_channels = c_in // len(self.windows_list)

        self.conv1 = nn.Conv2d(c_in, len(self.windows_list) * bottlenck_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(len(self.windows_list) * bottlenck_channels, len(self.windows_list) * bottlenck_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(len(self.windows_list) * bottlenck_channels, c_in, 3, 1, 1)
        self.position = nn.Conv2d(bottlenck_channels, bottlenck_channels, 3, 1, 1)

        # self.module = SELayer(bottlenck_channels, reduction=16)
        self.module = PSA(bottlenck_channels, bottlenck_channels)
        # self.module = Cbam(bottlenck_channels, 7)

    def _get_unfold_config(self, x):
        """
        计算不同尺寸输入下的unfold的核长kernel和步长stride，以便于能处理到每个像素
        """
        N, C, H, W = x.shape
        kernel_list = []
        stride_list = []

        for i in range(len(self.windows_list)):
            H_stride = H // self.windows_list[i]
            H_kernel = H_stride + H % self.windows_list[i]
            W_stride = W // self.windows_list[i]
            W_kernel = W_stride + W % self.windows_list[i]
            kernel_list.append((H_kernel, W_kernel))
            stride_list.append((H_stride, W_stride))

        return kernel_list, stride_list

    def forward(self, x):
        x = self.conv1(x)
        x_list = [i for i in torch.chunk(x, chunks=3, dim=1)]
        N, C, H, W = x_list[0].shape
        kernel_list, stride_list = self._get_unfold_config(x_list[0])
        # print("kernel_list: " + str(kernel_list))
        # print("stride_list: " + str(stride_list))

        # 多尺度unfold核展开
        out_list = []
        for i in range(len(self.windows_list)):
            # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
            # (N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)
            unfolded = F.unfold(x_list[i], kernel_size=kernel_list[i], stride=stride_list[i])
            # view为(N, C, self.kernel_list[i], self.kernel_list[i], num_windows),便于后续处理
            unfolded = unfolded.view(-1, C, kernel_list[i][0], kernel_list[i][1])

            # 输入注意力模块
            attention = self.module(unfolded) + self.position(unfolded)

            # 转化为(N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)，便于进行fold操作
            attention = attention.view(N, C * kernel_list[i][0] * kernel_list[i][1], self.windows_list[i] ** 2)

            # 计算重叠部分的权重
            count = torch.ones_like(attention)
            count = F.fold(count, output_size=(H, W), kernel_size=kernel_list[i], stride=stride_list[i])
            # print("count 形状: " + str(count.shape))

            # 重新通过 fold 将滑动窗口展开为完整的张量，形状为(N, C, H_orin, W_orin)
            attention = F.fold(attention, output_size=(H, W), kernel_size=kernel_list[i], stride=stride_list[i])
            # print("fold后的形状: " + str(output.shape))

            # 对重叠部分取平均值
            attention = attention / count
            out_list.append(attention)

        out = self.conv3(self.conv2(torch.cat(out_list, dim=1)) + x)
        return out


# 测试代码
if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(1, 576, 21, 13)

    module = DPConv(x.shape[1], 4)
    flops, params = profile(module, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
