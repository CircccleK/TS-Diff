import os
import os.path as osp
import sys
from collections import OrderedDict

import torch
import torchvision.utils as vutils
from torch.nn import functional as F

from archs import build_network
from archs.repnr_utils import build_repnr_arch_from_base
from data import LL_Dataset
from models.repdiff_base_model import RepDiffBaseModel
from utils import get_root_logger, imwrite, tensor2img, noise
from utils.raw_data import raw2rgb_batch
from utils.registry import MODEL_REGISTRY

sys.path.append('.')
import numpy as np
import cv2

cv2.setNumThreads(1)
from torch.nn.parallel import DataParallel, DistributedDataParallel


@MODEL_REGISTRY.register()
class RepDiffModel(RepDiffBaseModel):

    def __init__(self, opt):
        super(RepDiffModel, self).__init__(opt)
        self.best_val_loss = self.opt['val'].get('best_val_loss', -1e6)

        # define u-net network
        self.unet = build_network(opt['network_unet'])
        self.global_corrector = build_network(opt['network_global_corrector'])

        repnr_opt = self.opt.get('repnr_opt', None)
        if repnr_opt:
            self.generalize = True
            logger = get_root_logger()
            logger.info(f'Convert {self.unet._get_name()} into RepNRBase using kwargs:\n{repnr_opt}')
            self.unet = build_repnr_arch_from_base(self.unet, **repnr_opt)
            self.unet.pretrain()

        self.unet = self.model_to_device(self.unet)
        self.global_corrector = self.model_to_device(self.global_corrector)
        opt['network_ddpm']['denoise_fn'] = self.unet
        opt['network_ddpm']['color_fn'] = self.global_corrector

        self.ddpm = build_network(opt['network_ddpm'])
        self.ddpm = self.model_to_device(self.ddpm)

        if isinstance(self.unet, (DataParallel, DistributedDataParallel)):
            self.bare_unet = self.unet.module
        else:
            self.bare_unet = self.unet

        if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)):
            self.bare_model = self.ddpm.module
        else:
            self.bare_model = self.ddpm

        self.bare_model.set_new_noise_schedule(schedule_opt=opt['ddpm_schedule'],
                                               device=self.device)
        self.bare_model.set_loss(device=self.device)
        self.print_network(self.ddpm)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            if isinstance(param_key, list):
                self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_g', True), param_key[0])
                self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_g', True), param_key[1])
            else:
                self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.cur_camera = self.opt['datasets']['train']['camera']
        self.camera_id = None
        if self.cur_camera == "Virtual":
            self.noise_model = noise.NoiseModel(self.cur_camera)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.ddpm.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.ddpm.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, phase='train'):
        if self.cur_camera == "Virtual" and phase == 'train':
            self.camera_id = torch.randint(0, self.opt['datasets']['train']['virtual_camera_count'], (1,)).item()
            data['LR'] = self.noise_model(data['HR'], self.camera_id)
            data['LR'] = np.maximum(np.minimum(data['LR'], 1.0), 0)
            batch_size = data['HR'].shape[0]
            hr_process = []
            lq_process = []
            for idx in range(batch_size):
                data_process = LL_Dataset.process(data['LR'][idx], data['lq_path'][idx], data['HR'][idx], data['gt_path'][idx], data['position_encoding'][idx], self.opt['datasets']['train'])
                hr_process.append(data_process['HR'])
                lq_process.append(data_process['LR'])

            data['HR'] = torch.stack(hr_process, dim=0)
            data['LR'] = torch.stack(lq_process, dim=0)

        self.LR = data['LR'].to(self.device)
        self.HR = data['HR'].to(self.device)
        self.gt_path = data['gt_path']
        self.lq_path = data['lq_path']
        if 'pad_left' in data:
            self.pad_left = data['pad_left'].to(self.device)
            self.pad_right = data['pad_right'].to(self.device)
            self.pad_top = data['pad_top'].to(self.device)
            self.pad_bottom = data['pad_bottom'].to(self.device)

    def optimize_parameters(self, current_iter):
        # if self.opt['train'].get('mask_loss', False):
        #     assert self.opt['train'].get('cal_noise_only', False), "ma、sk_loss can only used with cal_noise_only, now"
        # optimize net_g
        self.optimizer_g.zero_grad()

        pred_noise, noise, x_recon_cs, x_start, t, color_scale = self.ddpm(self.HR, self.LR, self.camera_id,
                                                                           train_type=self.opt['train'].get(
                                                                               'train_type', None),
                                                                           different_t_in_one_batch=self.opt[
                                                                               'train'].get('different_t_in_one_batch',
                                                                                            None),
                                                                           t_sample_type=self.opt['train'].get(
                                                                               't_sample_type', None),
                                                                           pred_type=self.opt['train'].get('pred_type',
                                                                                                           None),
                                                                           clip_noise=self.opt['train'].get(
                                                                               'clip_noise', None),
                                                                           color_shift=self.opt['train'].get(
                                                                               'color_shift', None),
                                                                           color_shift_with_schedule=self.opt[
                                                                               'train'].get('color_shift_with_schedule',
                                                                                            None),
                                                                           t_range=self.opt['train'].get('t_range',
                                                                                                         None),
                                                                           cs_on_shift=self.opt['train'].get(
                                                                               'cs_on_shift', None),
                                                                           cs_shift_range=self.opt['train'].get(
                                                                               'cs_shift_range', None),
                                                                           t_border=self.opt['train'].get('t_border',
                                                                                                          None),
                                                                           down_uniform=self.opt['train'].get(
                                                                               'down_uniform', False),
                                                                           down_hw_split=self.opt['train'].get(
                                                                               'down_hw_split', False),
                                                                           pad_after_crop=self.opt['train'].get(
                                                                               'pad_after_crop', False),
                                                                           input_mode=self.opt['train'].get(
                                                                               'input_mode', None),
                                                                           divide=self.opt['train'].get('divide', None),
                                                                           frozen_denoise=self.opt['train'].get(
                                                                               'frozen_denoise', None),
                                                                           cs_independent=self.opt['train'].get(
                                                                               'cs_independent', None),
                                                                           shift_x_recon_detach=self.opt['train'].get(
                                                                               'shift_x_recon_detach', None))
        if self.opt['train'].get('vis_train', False) and current_iter <= self.opt['train'].get('vis_num', 10) and \
                self.opt['rank'] == 0:
            '''
            When the parameter 'vis_train' is set to True, the training process will be visualized. 
            The value of 'vis_num' corresponds to the number of visualizations to be generated.
            '''
            save_img_path = osp.join(self.opt['path']['visualization'], 'train')
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            save_img_path_train = osp.join(save_img_path, f'{current_iter}_noise_level_{self.bare_model.t}.png')
            save_img_path_lq = osp.join(save_img_path, f'{current_iter}_noise_level_{self.bare_model.t}_lq.png')

            if self.opt['train']['stage_in'] == 'raw':
                x_recon_print = raw2rgb_batch(self.gt_path, self.bare_model.x_recon, is_denormal=True, camera=self.opt['val'].get('camera', 'SonyA7S2'))
                noise_print = raw2rgb_batch(self.gt_path, self.bare_model.noise, is_denormal=True, camera=self.opt['val'].get('camera', 'SonyA7S2'))
                pred_noise_print = raw2rgb_batch(self.gt_path, self.bare_model.pred_noise, is_denormal=True, camera=self.opt['val'].get('camera', 'SonyA7S2'))
                x_start_print = raw2rgb_batch(self.gt_path, self.bare_model.x_start, is_denormal=True, camera=self.opt['val'].get('camera', 'SonyA7S2'))
                x_noisy_print = raw2rgb_batch(self.gt_path, self.bare_model.x_noisy, is_denormal=True, camera=self.opt['val'].get('camera', 'SonyA7S2'))
                x_lq = raw2rgb_batch(self.gt_path, self.LR[:, :4, :, :], is_denormal=True, camera=self.opt['val'].get('camera', 'SonyA7S2'))
                img_print = torch.cat([x_start_print, noise_print, x_noisy_print, pred_noise_print, x_recon_print], dim=0)
                vutils.save_image(x_lq.data, save_img_path_lq)
                vutils.save_image(img_print.data, save_img_path_train)
            else:
                x_recon_print = tensor2img(self.bare_model.x_recon, min_max=(-1, 1))
                noise_print = tensor2img(self.bare_model.noise, min_max=(-1, 1))
                pred_noise_print = tensor2img(self.bare_model.pred_noise, min_max=(-1, 1))
                x_start_print = tensor2img(self.bare_model.x_start, min_max=(-1, 1))
                x_noisy_print = tensor2img(self.bare_model.x_noisy, min_max=(-1, 1))
                img_print = np.concatenate([x_start_print, noise_print, x_noisy_print, pred_noise_print, x_recon_print],
                                       axis=0)
                imwrite(img_print, save_img_path_train)
                
        l_g_total = 0
        loss_dict = OrderedDict()

        l_g_x0 = F.l1_loss(x_recon_cs, x_start) * self.opt['train'].get('l_g_x0_w', 1.0)
        if self.opt['train'].get('gamma_limit_train', None) and color_scale <= self.opt['train'].get(
                'gamma_limit_train', None):
            l_g_x0 = l_g_x0 * 1e-12
        loss_dict['l_g_x0'] = l_g_x0
        l_g_total += l_g_x0

        if not self.opt['train'].get('frozen_denoise', False):
            l_g_noise = F.l1_loss(pred_noise, noise)
            loss_dict['l_g_noise'] = l_g_noise
            l_g_total += l_g_noise

        l_g_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)


    def save(self, epoch, current_iter, label=None):
        self.save_network([self.ddpm], 'net_g', current_iter, label, param_key=['params'])




