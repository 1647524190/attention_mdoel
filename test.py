import torch

# 假设输入的张量形状为 (N, C, H, W)
x = torch.randn(1, 2, 2, 2)
result = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
print("x: ", x)
print("result: ", result)
