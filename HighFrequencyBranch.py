import torch
import torch.nn as nn

"""
    A module branch for extracting high-frequency channels of input features.
    paper：RingMo-lite: A Remote Sensing Multi-task Lightweight Network with CNN-Transformer Hybrid Framework
"""


class HighFrequencyBranch(nn.Module):
    def __init__(self, in_channels):
        super(HighFrequencyBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)
        self.conv1_pool = nn.Conv2d(1, in_channels // 2, kernel_size=1)

    def forward(self, x):
        C = x.shape[1]
        front_half = x[:, :C // 2, :, :]
        print("front_half shape" + str(front_half.shape))
        back_half = x[:, C // 2:, :, :]
        print("back_half shape" + str(back_half.shape))

        front_half = self.conv1(front_half)
        front_half = self.conv3(front_half)

        back_half = torch.max(back_half, 1)[0].unsqueeze(1)
        print(back_half.shape)
        back_half = self.conv1_pool(back_half)

        return torch.cat((front_half, back_half), dim=1)


if __name__ == '__main__':
    input = torch.rand(4, 64, 256, 256)
    Block = HighFrequencyBranch(input.shape[1])
    out = Block(input)
    print(out.shape)
