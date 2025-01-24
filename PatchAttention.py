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
        return pos_embed.view(1, self.embed_dim, h, w)  # [1, C, H, W]


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

        self.pos_encoder = SinCosPositionalEncoding2D(self.cin)

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

    def _generate_position_encoding(self, H, W):
        """
        直接生成位置编码。对每个像素位置生成编码。
        Args:
            H: 特征图的高度
            W: 特征图的宽度
        Returns:
            pos_encoding: 位置编码张量，形状为 (1, C, H, W)
        """
        # 生成位置编码的相对位置
        position_h = torch.arange(0, H).unsqueeze(1).float()  # 形状 (H, 1)
        position_w = torch.arange(0, W).unsqueeze(0).float()  # 形状 (1, W)

        # 生成正弦/余弦编码
        div_term = torch.exp(
            torch.arange(0, self.cin, 2).float() * -(math.log(10000.0) / self.cin))  # div_term 控制频率
        pos_encoding = torch.zeros(H, W, self.cin)

        # 偶数维度为sin，奇数维度为cos
        pos_encoding[:, :, 0::2] = torch.sin(position_h.unsqueeze(1) * div_term)  # 对应高度方向的编码
        pos_encoding[:, :, 1::2] = torch.cos(position_h.unsqueeze(1) * div_term)  # 对应宽度方向的编码

        # 交换维度得到 (1, pos_dim, H, W)
        pos_encoding = pos_encoding.permute(2, 0, 1).unsqueeze(0)  # (1, pos_dim, H, W)
        return pos_encoding

    def forward(self, x):
        N, C, H, W = x.shape
        kernel_list, stride_list = self._get_unfold_config(x)

        expnsionx = self.expansion(x)
        listx = [i for i in torch.chunk(expnsionx, chunks=len(self.windows), dim=1)]

        # 生成位置编码，形状为 (1, C, H, W)
        pos = self.pos_encoder(H, W)

        # 多尺度unfold核展开
        out_list = []
        for i in range(len(self.windows)):
            # 对张量进行unfold展开，分解为 num_windows * num_windows 个子块
            # (N, C * kernel_list[i][0] * kernel_list[i][1], self.windows[i] ** 2)
            unfolded = F.unfold(listx[i], kernel_size=kernel_list[i], stride=stride_list[i])
            pos_unfold = F.unfold(pos, kernel_size=kernel_list[i], stride=stride_list[i])

            # view为(N, C, self.windows[i] ** 2, kernel_list[i][0] * kernel_list[i][1]),便于后续处理
            unfolded = unfolded.view(N, C, self.windows[i] ** 2, kernel_list[i][0] * kernel_list[i][1])
            pos_unfold = pos_unfold.view(1, C, self.windows[i] ** 2, kernel_list[i][0] * kernel_list[i][1])

            # 输入注意力模块
            attn = self.module(unfolded + pos_unfold)

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

        out = self.fusion(self.resume(torch.cat(out_list, dim=1) + expnsionx) + x)
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
