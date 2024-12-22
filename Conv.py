import torch.nn as nn


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
