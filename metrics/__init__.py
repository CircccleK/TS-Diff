from copy import deepcopy

from utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .psnr_pytorch import calculate_psnr_pytorch
from .ssim_pytorch import calculate_ssim_pytorch
from .psnr_ssim_skimagel import calculate_ssim_skimage, calculate_psnr_skimage
from .lpips_lol import calculate_lpips_lol

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_psnr_pytorch', \
           'calculate_ssim_pytorch', 'calculate_lpips_lol', 'calculate_ssim_skimage', 'calculate_psnr_skimage']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
