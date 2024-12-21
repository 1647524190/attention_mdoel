import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from thop import profile
from CBAM import Cbam
from PSA import PSA
from SEAttention import SELayer


class Conv(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=0, g=1, d=1, bn=True, act=False, init=True):

        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=g, dilation=d, bias=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.bn = nn.BatchNorm2d(cout) if bn is True else (bn if isinstance(bn, nn.Module) else nn.Identity())

        if init:
            self.initialize_weights()

    def initialize_weights(self):
        """初始化卷积层和批归一化层的权重。"""
        if isinstance(self.conv, nn.Conv2d):
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0)

        if isinstance(self.bn, nn.BatchNorm2d):
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DPConv(nn.Module):
    def __init__(self, in_channels, num_windows):
        """
        类卷积设计注意力模块。
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(DPConv, self).__init__()
        self.num_windows_list = [num_windows // 2, num_windows, num_windows + num_windows // 2]
        bottlenck_channels = in_channels // len(self.num_windows_list)

        self.conv1 = Conv(in_channels, len(self.num_windows_list) * bottlenck_channels, k=1, s=1)
        self.conv2 = Conv(len(self.num_windows_list) * bottlenck_channels, in_channels, k=3, s=1, p=1)
        self.conv3 = Conv(in_channels, in_channels, k=3, s=1, p=1)
        self.position = Conv(bottlenck_channels, bottlenck_channels, k=3, s=1, p=1)

        # self.attention = SELayer(bottlenck_channels, reduction=16)
        self.attention = PSA(bottlenck_channels, bottlenck_channels)
        # self.attention = Cbam(bottlenck_channels, 7)

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
        x = self.conv1(x)
        x_list = [i for i in torch.chunk(x, chunks=3, dim=1)]
        N, C, H, W = x_list[0].shape
        kernel_list, stride_list = self._get_unfold_config(x_list[0])
        # print("kernel_list: " + str(kernel_list))
        # print("stride_list: " + str(stride_list))

        # 多尺度unfold核展开
        out_list = []
        for i in range(len(self.num_windows_list)):
            # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
            # (N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)
            unfolded = F.unfold(x_list[i], kernel_size=kernel_list[i], stride=stride_list[i])
            # view为(N, C, self.kernel_list[i], self.kernel_list[i], num_windows),便于后续处理
            unfolded = unfolded.view(-1, C, kernel_list[i][0], kernel_list[i][1])

            # 输入注意力模块
            # attention_output = unfolded
            attention_output = self.attention(unfolded) + self.position(unfolded)

            # 转化为(N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)，便于进行fold操作
            attention_output = attention_output.view(N, C * kernel_list[i][0] * kernel_list[i][1],
                                                     self.num_windows_list[i] ** 2)

            # 计算重叠部分的权重
            count = F.fold(torch.ones_like(attention_output), output_size=(H, W), kernel_size=kernel_list[i],
                           stride=stride_list[i])
            # print("count 形状: " + str(count.shape))

            # 重新通过 fold 将滑动窗口展开为完整的张量，形状为(N, C, H_orin, W_orin)
            attention_output = F.fold(attention_output, output_size=(H, W), kernel_size=kernel_list[i],
                                      stride=stride_list[i])
            # print("fold后的形状: " + str(output.shape))

            # 对重叠部分取平均值
            attention_output = attention_output / count
            out_list.append(attention_output)

        out = self.conv3(self.conv2(torch.cat(out_list, dim=1)) + x)
        return out


# 测试代码
if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(1, 576, 21, 13)

    attention = DPConv(x.shape[1], 4)
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
