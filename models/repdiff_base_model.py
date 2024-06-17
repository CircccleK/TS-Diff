import os
import os.path as osp
import sys

import torch
import torchvision.utils as vutils
from torch.nn import functional as F
from tqdm import tqdm

from metrics import calculate_metric
from models.base_model import BaseModel
from utils import get_root_logger, imwrite, tensor2img
from utils.img_util import denormalize
from utils.raw_data import raw2rgb_batch

sys.path.append('.')
import numpy as np


class RepDiffBaseModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.generalize = False
    def test(self):
        if self.opt['val'].get('test_speed', False):
            assert self.opt['val'].get('ddim_pyramid', False), "please use ddim_pyramid"
            with torch.no_grad():
                iterations = self.opt['val'].get('iterations', 100)
                input_size = self.opt['val'].get('input_size', [400, 600])

                LR = torch.randn(1, 8, input_size[0], input_size[1]).to(self.device)
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                self.bare_model.denoise_fn.eval()

                # GPU warm up
                print('GPU warm up')
                for _ in tqdm(range(50)):
                    self.output = self.bare_model.sample(LR,
                                                         sample_type=self.opt['val']['sample_type'],
                                                         pyramid_list=self.opt['val'].get('pyramid_list'),
                                                         continous=self.opt['val'].get('ret_process',
                                                                                       False),
                                                         ddim_timesteps=self.opt['ddpm_schedule'].get(
                                                             'sample_timesteps', 50),
                                                         return_pred_noise=self.opt['val'].get(
                                                             'return_pred_noise', False),
                                                         return_x_recon=self.opt['val'].get('ret_x_recon',
                                                                                            False),
                                                         ddim_discr_method=self.opt['val'].get(
                                                             'ddim_discr_method', 'uniform'),
                                                         ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                         pred_type=self.opt['val'].get('pred_type',
                                                                                       'noise'),
                                                         clip_noise=self.opt['val'].get('clip_noise',
                                                                                        False),
                                                         save_noise=self.opt['val'].get('save_noise',
                                                                                        False),
                                                         color_gamma=self.opt['val'].get('color_gamma',
                                                                                         None),
                                                         color_times=self.opt['val'].get('color_times', 1),
                                                         return_all=self.opt['val'].get('ret_all', False))

                # speed test
                times = torch.zeros(iterations)  # Store the time of each iteration
                for iter in tqdm(range(iterations)):
                    starter.record()
                    self.output = self.bare_model.sample(LR,
                                                         sample_type=self.opt['val']['sample_type'],
                                                         pyramid_list=self.opt['val'].get('pyramid_list'),
                                                         continous=self.opt['val'].get('ret_process',
                                                                                       False),
                                                         ddim_timesteps=self.opt['ddpm_schedule'].get(
                                                             'sample_timesteps', 50),
                                                         return_pred_noise=self.opt['val'].get(
                                                             'return_pred_noise', False),
                                                         return_x_recon=self.opt['val'].get('ret_x_recon',
                                                                                            False),
                                                         ddim_discr_method=self.opt['val'].get(
                                                             'ddim_discr_method', 'uniform'),
                                                         ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                         pred_type=self.opt['val'].get('pred_type',
                                                                                       'noise'),
                                                         clip_noise=self.opt['val'].get('clip_noise',
                                                                                        False),
                                                         save_noise=self.opt['val'].get('save_noise',
                                                                                        False),
                                                         color_gamma=self.opt['val'].get('color_gamma',
                                                                                         None),
                                                         color_times=self.opt['val'].get('color_times', 1),
                                                         return_all=self.opt['val'].get('ret_all', False))
                    ender.record()
                    # Synchronize GPU
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    times[iter] = curr_time
                    # print(curr_time)

                mean_time = times.mean().item()
                logger = get_root_logger()
                logger.info("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))
                import sys
                sys.exit()
        with torch.no_grad():
            self.bare_model.denoise_fn.eval()
            self.output = self.bare_model.sample(self.LR,
                                                 sample_type=self.opt['val']['sample_type'],
                                                 pyramid_list=self.opt['val'].get('pyramid_list'),
                                                 continous=self.opt['val'].get('ret_process', False),
                                                 ddim_timesteps=self.opt['ddpm_schedule'].get(
                                                     'sample_timesteps', 50),
                                                 return_pred_noise=self.opt['val'].get('return_pred_noise',
                                                                                       False),
                                                 return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                 ddim_discr_method=self.opt['val'].get('ddim_discr_method',
                                                                                       'uniform'),
                                                 ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                 pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                 clip_noise=self.opt['val'].get('clip_noise', False),
                                                 save_noise=self.opt['val'].get('save_noise', False),
                                                 color_gamma=self.opt['val'].get('color_gamma', None),
                                                 color_times=self.opt['val'].get('color_times', 1),
                                                 return_all=self.opt['val'].get('ret_all', False),
                                                 fine_diffV2=self.opt['val'].get('fine_diffV2', False),
                                                 fine_diffV2_st=self.opt['val'].get('fine_diffV2_st', 200),
                                                 fine_diffV2_num_timesteps=self.opt['val'].get(
                                                     'fine_diffV2_num_timesteps', 20),
                                                 do_some_global_deg=self.opt['val'].get(
                                                     'do_some_global_deg', False),
                                                 use_up_v2=self.opt['val'].get('use_up_v2', False))
            self.bare_model.denoise_fn.train()

            if hasattr(self, 'pad_left') and not self.opt['val'].get('ret_process', False):
                self.output = pad_tensor_back(self.output, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.LR = pad_tensor_back(self.LR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.HR = pad_tensor_back(self.HR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
           self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        logger = get_root_logger()
        if self.opt['val'].get('fix_seed', False):
            next_seed = np.random.randint(10000000)
            logger.info(f'next_seed={next_seed}')

        # finetune
        if hasattr(self.bare_unet, 'generalize') and self.generalize:
            self.bare_unet.generalize()


        if self.opt['val'].get('save_img', False) and self.opt['val'].get('cal_score', False):
            self.save_images(dataloader, current_iter)
            return

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}


        metric_data = dict()

        pbar = tqdm(total=len(dataloader), unit='image')
        cnt = 0
        for idx, val_data in enumerate(dataloader):
            if self.opt['val'].get('fix_seed', False):
                from utils import set_random_seed
                set_random_seed(0)
            if not self.opt['val'].get('cal_all', False) and \
                    not self.opt['val'].get('cal_score', False) and \
                    int(self.opt['ddpm_schedule']['n_timestep']) >= 4 and idx >= 3:
                break
            if 'ELD' in self.opt['name']:
                parts = val_data['lq_path'][0].split('/')
                img_name = parts[-2] + "_" + osp.splitext(parts[-1])[0]
            else:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data, phase='val')
            self.test()

            sr_img, gt_img, lq_img = self.get_current_visuals()

            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metrics = calculate_metric(metric_data, opt_)
                    if self.opt['val'].get('cal_score', False):
                        logger.info(f'img: {os.path.basename(self.lq_path[0])}, {name}: {metrics}')
                        metric_data['img'] = lq_img
                        metrics_lq = calculate_metric(metric_data, opt_)
                        metric_data['img'] = sr_img
                        logger.info(f'lq: {os.path.basename(self.lq_path[0])}, {name}: {metrics_lq}')
                    self.metric_results[name] += metrics

            # tentative for out of GPU memory
            del self.LR
            del self.output
            torch.cuda.empty_cache()
            pbar.update(1)

            pbar.set_description(f'Test {img_name}')
            if self.opt['val'].get('cal_score_num', None):
                if idx >= self.opt['val'].get('cal_score_num', None):
                    break
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1 - cnt)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        if self.metric_results['psnr'] > self.best_val_loss:  # larger value indicates better
            self.best_val_loss = self.metric_results['psnr']
            logger.info(
                f'saving the best model at the end of , iters {current_iter}, best_val_loss {self.best_val_loss}')
            self.save(0, 0, label='best')
            self.save_training_state(0, 0, label='best')
            if save_img:
                self.save_images(dataloader, current_iter)
        if self.opt['val'].get('cal_score', False):
            self.best_val_loss = self.opt['val'].get('best_val_loss', -1e6)
        if self.opt['val'].get('fix_seed', False):
            from utils import set_random_seed
            set_random_seed(next_seed)

    def save_images(self, dataloader, current_iter):
        pbar = tqdm(total=len(dataloader), unit='image')
        for idx, val_data in enumerate(dataloader):
            if 'ELD' in self.opt['name']:
                parts = val_data['lq_path'][0].split('/')
                img_name = parts[-2] + "_" + osp.splitext(parts[-1])[0]
            else:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, phase='val')
            self.test()

            if not self.opt['val'].get('ret_process', False):
                sr_img, gt_img, lq_img = self.get_current_visuals(save_image=True)

            if idx < self.opt['val'].get('show_num', 3) or self.opt['val'].get('show_all', False):
                save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                         f'{img_name}_{current_iter}.png')
                if not self.opt['val'].get('ret_process', False):
                    if self.opt['val'].get('cal_score', False):
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 f'{current_iter}_{img_name}_sr.png')
                        imwrite(sr_img, save_img_path)  # 保存单个图片
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 f'{current_iter}_{img_name}_lq.png')
                        imwrite(lq_img, save_img_path)
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 f'{current_iter}_{img_name}_gt.png')
                        imwrite(gt_img, save_img_path)
                    else:
                        imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)
                else:
                    sample_inter, _, _, _ = self.output.shape
                    for i in range(sample_inter):
                        output_rgb = raw2rgb_batch(self.lq_path, self.output[i].unsqueeze(0), True)
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 f'{current_iter}_{img_name}_{i}.png')
                        vutils.save_image(output_rgb.data, save_img_path)
            del self.LR
            del self.output
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Save {img_name}')
            if self.opt['val'].get('cal_score_num', None):
                if idx >= self.opt['val'].get('cal_score_num', None):
                    break
        pbar.close()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        logger = get_root_logger()
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self, save_image=False):
        if self.LR.shape != self.output.shape:
            self.LR = F.interpolate(self.LR, self.output.shape[2:])
            self.HR = F.interpolate(self.HR, self.output.shape[2:])

        # 归一化[0, 1]
        self.output = denormalize(self.output)
        self.HR = denormalize(self.HR)
        self.LR = denormalize(self.LR[:, :self.opt['network_ddpm']['channels'], :, :])

        # 亮度恢复
        if self.opt['datasets']['val']['stage_in'] == 'raw':
            self.output = self.illuminance_correct(self.output, self.HR)

        if save_image and self.opt['datasets']['train']['stage_in'] == 'raw':
            self.HR = raw2rgb_batch(self.gt_path, self.HR, camera=self.opt['val'].get('camera', 'SonyA7S2'))
            self.output = raw2rgb_batch(self.gt_path, self.output, camera=self.opt['val'].get('camera', 'SonyA7S2'), save_image_L118=True)
            self.LR = raw2rgb_batch(self.gt_path, self.LR, camera=self.opt['val'].get('camera', 'SonyA7S2'))

        gt_img = self.HR.detach().cpu()
        sr_img = self.output.detach().cpu()
        lq_img = self.LR.detach().cpu()
        sr_img = tensor2img(sr_img)
        gt_img = tensor2img(gt_img)
        lq_img = tensor2img(lq_img)

        if self.opt['datasets']['val']['stage_in'] == 'srgb' and self.opt['val'].get('use_kind_align', False):
            '''
            References:
            https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py
            https://github.com/wyf0912/LLFlow/blob/main/code/test.py
            '''
            gt_mean = np.mean(gt_img)
            sr_mean = np.mean(sr_img)
            sr_img = sr_img * gt_mean / sr_mean # 调整亮度，观察效果
            sr_img = np.clip(sr_img, 0, 255)

        if save_image:
            # 计算pnsr时如果为uint8的话，会因为精度问题掉点，所以计算完psnr再转换
            sr_img = sr_img.astype(np.uint8)
            gt_img = gt_img.astype(np.uint8)
            lq_img = lq_img.astype(np.uint8)

        return sr_img, gt_img, lq_img
    
    def illuminance_correct(self,  predict, source):
        N, C, H, W = predict.shape
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]

        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)
        output = num / den * predict
        # print(num / den)

        return output
