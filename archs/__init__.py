from copy import deepcopy

from archs import ddpm_arch, gc_arch, sr3unet_arch, vgg_arch
from utils import get_root_logger
from utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
