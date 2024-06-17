import argparse
import random
from collections import OrderedDict
from os import path as osp

import torch
import yaml

from utils import set_random_seed
from utils.dist_util import get_dist_info, init_dist, master_only


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = osp.join(*osp.split(root_path), 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
        if opt['datasets']['train']['stage_in'] == 'raw' and opt['datasets']['train']['save_syn_data']:
            opt['datasets']['train']['syn_path'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 2
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 4
            opt['datasets']['val']['fns_root'] = './data/Sony/Sony_val_debug.txt'
    else:  # test
        results_root = osp.join(*osp.split(root_path), 'experiments', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['models'] = osp.join(results_root, 'models')
        opt['path']['log'] = results_root
        opt['path']['training_states'] = osp.join(results_root, 'training_states')
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    if opt['datasets']['train']['stage_in'] == 'raw':
        if opt['datasets']['train']['concat_with_position_encoding'] and opt['datasets']['train']['concat_with_hiseq']:
            opt['network_unet']['in_channel'] = 15
        elif not opt['datasets']['train']['concat_with_position_encoding'] and opt['datasets']['train']['concat_with_hiseq']:
            opt['network_unet']['in_channel'] = 11
        elif opt['datasets']['train']['concat_with_position_encoding'] and not opt['datasets']['train']['concat_with_hiseq']:
            opt['network_unet']['in_channel'] = 12
        else:
            opt['network_unet']['in_channel'] = 8
        opt['network_unet']['out_channel'] = 4
        opt['network_ddpm']['channels'] = 4
        opt['network_global_corrector']['in_nc'] = 4
        opt['network_global_corrector']['out_nc'] = 4
    else:
        opt['network_unet']['in_channel'] = 13
        opt['network_unet']['out_channel'] = 3
        opt['network_ddpm']['channels'] = 3
        opt['network_global_corrector']['in_nc'] = 3
        opt['network_global_corrector']['out_nc'] = 3

    # change some options for debug mode
    if 'ELD' in opt['name']:
        opt['datasets']['train']['scenes'] = list(range(1, 10 + 1))
        opt['datasets']['val']['scenes'] = list(range(1, 10 + 1))
        if opt['camera_idx'] == 1:
            opt['datasets']['train']['camera'] = 'CanonEOS70D'
            opt['datasets']['val']['camera'] = 'CanonEOS70D'
            opt['datasets']['train']['fns_root'] = './data/ELD/ELD_train_CanonEOS70D.txt'
            if 'test' not in opt['name']:
                opt['datasets']['val']['fns_root'] = './data/ELD/ELD_val_CanonEOS70D.txt'
            else:
                opt['datasets']['val']['fns_root'] = ['./data/ELD/CanonEOS70D_val_ratio_100_ELD.txt',
                                                      './data/ELD/CanonEOS70D_val_ratio_200_ELD.txt']
        elif opt['camera_idx'] == 2:
            opt['datasets']['train']['camera'] = 'NikonD850'
            opt['datasets']['val']['camera'] = 'NikonD850'
            opt['datasets']['train']['fns_root'] = './data/ELD/ELD_train_NikonD850.txt'
            opt['datasets']['val']['fns_root'] = './data/ELD/ELD_val_NikonD850.txt'
            if 'test' not in opt['name']:
                opt['datasets']['val']['fns_root'] = './data/ELD/ELD_val_NikonD850.txt'
            else:
                opt['datasets']['val']['fns_root'] = ['./data/ELD/NikonD850_val_ratio_100_ELD.txt',
                                                      './data/ELD/NikonD850_val_ratio_200_ELD.txt']
        elif opt['camera_idx'] == 3:
            opt['datasets']['train']['camera'] = 'SonyA7S2'
            opt['datasets']['val']['camera'] = 'SonyA7S2'
            opt['datasets']['train']['fns_root'] = './data/ELD/ELD_train_SonyA7S2.txt'
            opt['datasets']['val']['fns_root'] = './data/ELD/ELD_val_SonyA7S2.txt'
            if 'test' not in opt['name']:
                opt['datasets']['val']['fns_root'] = './data/ELD/ELD_val_SonyA7S2.txt'
            else:
                opt['datasets']['val']['fns_root'] = ['./data/ELD/SonyA7S2_val_ratio_100_ELD.txt',
                                                      './data/ELD/SonyA7S2_val_ratio_200_ELD.txt']
        elif opt['camera_idx'] == 4:
            opt['datasets']['train']['camera'] = 'CanonEOS700D'
            opt['datasets']['val']['camera'] = 'CanonEOS700D'
            opt['datasets']['train']['fns_root'] = './data/ELD/ELD_train_CanonEOS700D.txt'
            opt['datasets']['val']['fns_root'] = './data/ELD/ELD_val_CanonEOS700D.txt'
            if 'test' not in opt['name']:
                opt['datasets']['val']['fns_root'] = './data/ELD/ELD_val_CanonEOS700D.txt'
            else:
                opt['datasets']['val']['fns_root'] = ['./data/ELD/CanonEOS700D_val_ratio_100_ELD.txt',
                                                      './data/ELD/CanonEOS700D_val_ratio_200_ELD.txt']

    return opt, args


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)
