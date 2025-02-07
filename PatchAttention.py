import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile
from block import PSA, CBAM, SELayer, GlobalContextBlock


class MaskFusion(nn.Module):
    def __init__(self, threshold=0.5):
        """
        threshold: 原始张量与特征张量相似度二值化阈值，默认0.5
        """
        super(MaskFusion, self).__init__()
        self.threshold = threshold
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=1, stride=1)

    def c_pool(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

    def forward(self, orin, feature):
        """
        基于同位置特征相似度的二值化掩模融合
        Args:
            orin (Tensor): 输入原始张量，形状为[N, C, H, W]
            feature (Tensor): 输入特征张量，形状与orin相同
        Returns:
            fused (Tensor): 融合结果张量，形状与输入一致
        """
        # 形状一致性检查
        assert orin.shape == feature.shape, f"Shape mismatch: orin {orin.shape}, feature {feature.shape}"

        conv_orin = self.conv1(self.c_pool(orin))
        conv_feature = self.conv2(self.c_pool(feature))

        # 计算同位置特征点积相似度
        similarity = torch.sum(conv_orin * conv_feature, dim=1, keepdim=True)  # 计算相似度，沿通道维度
        similarity = torch.sigmoid(similarity)  # 映射到0-1范围

        # 生成二值化掩模
        mask = (similarity > self.threshold).float()

        # 空间对齐融合
        fusion = mask * feature + (1 - mask) * orin
        return fusion


class PatchAttention(nn.Module):
    def __init__(self, cin, windows=None):
        """
        现有注意力模块均针对全局输入特征进行建模，而忽视区域特征建模。该模块旨在将特征图unfold为patch之后逐个对patch进行注意力建模，
        并之后重新fold为原特征图。从而实现注意力模块对局部区域的特征建模，同时采用多尺度unfold展开实现不同区域大小的特征建模。
        Args:
            cin: 输入通道数
            windows: unfold展开的窗口数列表
        """
        super(PatchAttention, self).__init__()
        # 检查 windows 是否为列表
        if windows is None:
            windows = [2, 4, 6]
        if not isinstance(windows, list):
            raise TypeError(f"Error: 'windows' must be a list, but got {type(windows).__name__}.")

        self.windows = windows
        self.cin = cin

        self.expansion = nn.Conv2d(self.cin, len(self.windows) * self.cin, kernel_size=1, stride=1)
        self.resume = nn.Conv2d(len(self.windows) * self.cin, self.cin, kernel_size=1, stride=1)
        self.fusion = nn.Conv2d(self.cin, self.cin, kernel_size=1, stride=1)

        self.maskfusion1 = MaskFusion()
        self.maskfusion2 = MaskFusion()

        # self.module = SELayer(self.cin, reduction=16)
        self.module = PSA(self.cin, self.cin)
        # self.module = CBAM(self.cin, 7)
        # self.module = GlobalContextBlock(self.cin)

    def _get_unfold_config(self, x):
        """
        计算不同展开patch下的unfold的核长kernel和步长stride
        Args:
            x: 输入张量
        Returns:
            kernel_list: 对应展开patch的kernel列表
            stride_list: 对应展开patch的stride列表
        """
        N, C, H, W = x.shape
        windows = torch.tensor(self.windows, dtype=torch.int32)  # 将 windows 转为张量

        # 计算步长（stride）
        H_stride = torch.div(H, windows, rounding_mode='floor')
        W_stride = torch.div(W, windows, rounding_mode='floor')

        # 计算核长 (kernel)
        H_kernel = H_stride + H % windows
        W_kernel = W_stride + W % windows

        # 将结果转换为列表形式
        kernel_list = list(zip(H_kernel.tolist(), W_kernel.tolist()))
        stride_list = list(zip(H_stride.tolist(), W_stride.tolist()))

        return kernel_list, stride_list

    def forward(self, x):
        N, C, H, W = x.shape
        kernel_list, stride_list = self._get_unfold_config(x)

        expnsionx = self.expansion(x)
        listx = [i for i in torch.chunk(expnsionx, chunks=len(self.windows), dim=1)]

        # 多尺度unfold核展开
        out_list = []
        for i in range(len(self.windows)):
            # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
            # (N, C * kernel_list[i][0] * kernel_list[i][1], self.windows[i] ** 2)
            unfolded = F.unfold(listx[i], kernel_size=kernel_list[i], stride=stride_list[i])

            # view为(N, C, self.windows[i] ** 2, kernel_list[i][0] * kernel_list[i][1]),便于后续处理
            unfolded = unfolded.view(N, C, self.windows[i] ** 2, kernel_list[i][0] * kernel_list[i][1])

            # 输入注意力模块
            attn = self.module(unfolded)

            # 转化为(N, C * self.kernel_list[i] * self.kernel_list[i], num_windows)，便于进行fold操作
            attn = attn.view(N, C * kernel_list[i][0] * kernel_list[i][1], self.windows[i] ** 2)

            # 计算重叠部分的权重
            count = torch.ones_like(attn)
            count = F.fold(count, output_size=(H, W), kernel_size=kernel_list[i], stride=stride_list[i])

            # 重新通过 fold 将滑动窗口展开为完整的张量，形状为(N, C, H_orin, W_orin)
            attn = F.fold(attn, output_size=(H, W), kernel_size=kernel_list[i], stride=stride_list[i])

            # 对重叠部分取平均值
            attn = attn / count
            out_list.append(attn)

        out = self.maskfusion1(torch.cat(out_list, dim=1), expnsionx)
        out = self.resume(out)
        out = self.maskfusion2(out, x)
        out = self.fusion(out)
        return out


# 测试代码
if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(4, 576, 21, 13)

    module = PatchAttention(x.shape[1], [1, 3, 5])
    flops, params = profile(module, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
