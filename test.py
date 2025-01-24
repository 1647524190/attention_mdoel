import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile
from block import PSA, CBAM, SELayer, GlobalContextBlock


class SinCosPositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, temperature=10000.):
        """
        2D 正余弦位置编码
        :param embed_dim: 编码维度
        :param temperature: 控制编码频率的温度参数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, h, w, device='cpu'):
        """
        生成位置编码
        :return: 位置编码，形状 [1, H*W, C]
        """
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1. / (self.temperature ** omega)

        out_w = torch.einsum('i,j->ij', grid_w.flatten(), omega)  # [H*W, pos_dim]
        out_h = torch.einsum('i,j->ij', grid_h.flatten(), omega)

        pos_embed = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
        return pos_embed.view(1, self.embed_dim, h, w)  # [1, H*W, C]


if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(1, 576, 21, 13)

    module = SinCosPositionalEncoding2D(256)
    pos = module.forward(20, 20)
