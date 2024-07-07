import torch
from torch import nn
from thop import profile

# 通道注意力机制
class channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention, self).__init__()
        # 最大池化贺平均池化，输出层的高贺宽都是1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        # 两次全连接
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获得输入的x的batch，通道数，高和宽
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.ave_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        ave_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + ave_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


# 空间注意力机制
class spacial_attention(nn.Module):
    def __init__(self, kernel=7):
        super(spacial_attention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel, 1, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 寻找通道上所有特征点的最大值和平均值
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        # 链接
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        # 相当于获得每个特征点的权值
        out = self.sigmoid(out)
        return out * x


class Cbam(nn.Module):
    def __init__(self, channel, kernel=7, ratio=16):
        super(Cbam, self).__init__()
        self.channel_attention = channel_attention(channel, ratio)
        self.spacial_attention = spacial_attention(kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spacial_attention(x)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 256, 762, 524)
    attention = Cbam(x.shape[1])
    x = attention(x)
    print('output shape:'+str(x.shape))
    flops, params = profile(attention, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))