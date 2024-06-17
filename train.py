import datetime
import logging
import os
import time
from os import path as osp

import math
import numpy as np
import torch
import torchvision.utils as vutils

from data import build_dataloader, build_dataset
from data.data_sampler import EnlargedSampler
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models import build_model
from utils import (AvgTimer, MessageLogger, check_resume, get_root_logger, get_time_str,
                   init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir, get_env_info)
from utils.options import copy_opt_file, dict2str, parse_options
from utils.raw_data import raw2rgb_batch


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

def setTotalIterAndMilesStone(opt, train_set_size):
    if 'finetune' in opt['name']:
        return
    total_epoch = int(opt['train']['total_epoch'])
    batch_size = opt['datasets']['train']['batch_size_per_gpu'] * opt['num_gpu']
    opt['train']['total_iter'] = math.ceil(total_epoch * train_set_size / batch_size)
    epoch_milestones = opt['train']['scheduler']['epoch_milestones']
    opt['train']['scheduler']['milestones'] = epoch_milestones
    # valid_iter_milestones = []
    # for valid_epoch_milestone in opt['val']['valid_epoch_milestones']:
    #     valid_iter_milestones.append(math.ceil(valid_epoch_milestone * train_set_size / batch_size))
    # opt['val']['valid_iter_milestones'] = valid_iter_milestones
    del opt['train']['scheduler']['epoch_milestones']
    if 'debug' not in opt['name']:
        # opt['val']['val_freq'] = math.ceil(opt['val']['val_freq_num'] / batch_size)
        opt['logger']['print_freq'] = math.ceil(opt['logger']['print_epoch'] * train_set_size / batch_size)
        # opt['logger']['save_checkpoint_freq'] = math.ceil(opt['logger']['save_checkpoint_epoch'] * train_set_size / batch_size)
    # del opt['val']['val_freq_num']
    del opt['logger']['print_epoch']
    # del opt['logger']['save_checkpoint_epoch']

def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            # cal total_iter & milestone
            setTotalIterAndMilesStone(opt, len(train_set))
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            if opt['train']['total_epoch'] is not None:
                total_epochs = int(opt['train']['total_epoch'])
            else:
                total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def crop_img(input_img):
    crop_size = 1424
    _, H, W = input_img.shape
    h = H // 2 - (crop_size // 2)
    w = W // 2 - (crop_size // 2)
    input_img = input_img[:, h: h + crop_size, w: w + crop_size]
    return input_img

def save_syn_img(opt, epoch, train_data):
    gt_path = train_data['gt_path']
    lq_path = train_data['lq_path']
    save_img_path = osp.join(opt['datasets']['train']['syn_path'], f'syn_train_{epoch}')
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    syn_lq = train_data['syn_raw']
    real_lq = train_data['real_raw']
    if opt['datasets']['train'].get('save_syn_rgb', False):
        syn_lq = raw2rgb_batch(gt_path, syn_lq, camera=opt['datasets']['train'].get('camera', 'SonyA7S2'))  # input_syn
        # real_lq = raw2rgb_batch(gt_path, real_lq)
        batch_size, _, _, _ = syn_lq.shape
        for i in range(batch_size):
            img_print = crop_img(syn_lq[i])
            file_name = osp.basename(lq_path[i]).rsplit('.', 1)[0]
            save_img_path = osp.join(save_img_path, f'{file_name}.png')
            vutils.save_image(img_print, save_img_path)
            save_img_path = ''
    else:
        batch_size, _, _, _ = syn_lq.shape
        for i in range(batch_size):
            file_name = osp.basename(lq_path[i]).rsplit('.', 1)[0]
            img_path = osp.join(save_img_path, f'{file_name}')
            np.save(img_path, syn_lq[i].numpy())

def train_pipeline(root_path):
    global camera_id

    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='root_logger', log_level=logging.INFO, log_file=log_file)

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    logger.info(get_env_info())
    logger.info(dict2str(opt))

    assert opt['datasets']['train']['stage_in'] == opt['datasets']['val']['stage_in'], "The both value of stage_in is " \
                                                                                       "the same"

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        # current_iter = resume_state['iter']
        current_iter = math.ceil(resume_state['epoch'] * len(train_sampler.dataset) / (opt['datasets']['train']['batch_size_per_gpu'] * opt['num_gpu']))  # 针对batch不一致的时候rusume
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    cur_epoch_idx = 0

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break

            if opt['datasets']['train'].get('save_syn_data', False):
                if epoch < opt['datasets']['train'].get('train_syn_save_num', 5):
                    save_syn_img(opt, epoch, train_data)
                    train_data = prefetcher.next()
                    continue
                else:
                    return

            # training
            if not opt['val'].get('cal_score', False):
                model.feed_data(train_data)
                model.optimize_parameters(current_iter)
                iter_timer.record()
                if current_iter == 1:
                    # reset start time in msg_logger for more accurate eta_time
                    # not work in resume mode
                    msg_logger.reset_start_time()
                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

        # validation
        if opt['val'].get('cal_score', False) or opt.get('val') is not None and (epoch % opt['val']['val_freq_epoch'] == 0):
            if len(val_loaders) > 1:
                logger.warning('Multiple validation datasets are *only* supported by SRModel.')
            for val_loader in val_loaders:
                model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

        # update learning rate  # 重新启动时可能最开始几个epoch中val_freq_epoch不会按照预期
        model.update_learning_rate(epoch, warmup_iter=opt['train'].get('warmup_iter', -1))
        if epoch > int(opt['val']['valid_epoch_milestones'][cur_epoch_idx]):
            opt['val']['val_freq_epoch'] = math.ceil(opt['val']['val_freq_epoch'] * opt['val']['val_gamma'])
            cur_epoch_idx += 1

        # save models and training states
        if epoch % opt['logger']['save_checkpoint_epoch'] == 0:
            logger.info('Saving models and training states.')
            model.save(epoch, current_iter)
            model.save_training_state(epoch, current_iter)

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(osp.abspath(__file__), osp.pardir))
    train_pipeline(root_path)

