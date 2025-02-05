import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile

"""
    对于特征张量x与特征张量y，如何进行有效的特征融合。常规操作为（x+y），即简单通过特征相加来进行特征融合。
    再次我们提出问题：检查的特征相加无法实现高效的特征融合。
"""


class MaskFusion(nn.Module):
    def __init__(self, threshold=0.5):
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
            orin (Tensor): 输入张量1，形状为[N, C, H, W]
            feature (Tensor): 输入张量2，形状与orin相同
            threshold (float): 二值化阈值，默认0.5
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
        fused = mask * feature + (1 - mask) * orin
        return fused


if __name__ == '__main__':
    # 测试数据（batch=1, channel=3, height=5, width=5）
    orin = torch.randn(1, 2, 2, 3)
    feature = torch.randn(1, 2, 2, 3)

    # 执行融合
    fusion = MaskFusion()
    out = fusion(orin, feature)
