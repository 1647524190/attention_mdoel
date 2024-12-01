import torch.nn as nn


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, group=1, dilation=1, bias=True,
                 act=False):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding, groups=group,
                              dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
