import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile

"""
    对于特征张量x与特征张量y，如何进行有效的特征融合。常规操作为（x+y），即简单通过特征相加来进行特征融合。
    再次我们提出问题：检查的特征相加无法实现高效的特征融合。
"""


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
