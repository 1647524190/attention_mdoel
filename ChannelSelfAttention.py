import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSelfAttention, self).__init__()


if __name__ == '__main__':
    # x = torch.randn(2, 2, 4, 4)
    # x = torch.randn(4, 256, 762, 524)
    x = torch.randn(1, 576, 8, 8)

    attention = ChannelSelfAttention(x.shape[1])
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
