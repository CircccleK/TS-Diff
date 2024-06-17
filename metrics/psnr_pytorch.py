import torch
from utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_psnr_pytorch(img, img2): # 使用 PyTorch 框架，输入为张量（torch.Tensor）
    mse = (torch.abs(img - img2) ** 2).mean()
    psnr = 10 * torch.log10(1 * 1 / mse)
    return psnr