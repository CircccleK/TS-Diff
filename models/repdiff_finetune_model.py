import sys
from collections import OrderedDict
from copy import deepcopy

from torch.nn import functional as F

from archs import build_network
from archs.repnr_utils import build_repnr_arch_from_base, RepNRBase
from models import lr_scheduler
from models.repdiff_base_model import RepDiffBaseModel
from utils import get_root_logger
from utils.registry import MODEL_REGISTRY

sys.path.append('.')

from torch.nn.parallel import DataParallel, DistributedDataParallel


@MODEL_REGISTRY.register()
class RepDiffFinetuneModel(RepDiffBaseModel):
    def __init__(self, opt):
        super(RepDiffFinetuneModel, self).__init__(opt)

        self.best_val_loss = self.opt['val'].get('best_val_loss', -1e6)

        self.unet = build_network(opt['network_unet'])
        self.global_corrector = build_network(opt['network_global_corrector'])

        repnr_opt = opt['repnr_opt']
        self.repnr_opt = deepcopy(repnr_opt)
        logger = get_root_logger()
        logger.info(f'Convert {self.unet._get_name()} into RepNRBase using kwargs:\n{repnr_opt}')
        self.unet = build_repnr_arch_from_base(self.unet, **repnr_opt)

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
            self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        else:
            raise NotImplementedError(self.__name__ + ' is only for fintunning, please specify a pretrained path')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.ddpm.train()

        train_opt = self.opt['train']
        self.total_iter = train_opt['total_iter']
        self.align_iter = train_opt['align_iter']
        self.oomn_iter = train_opt['oomn_iter']
        assert (self.oomn_iter + self.align_iter) == self.total_iter

        # 检查对象 self.unet 是否有一个名为 'generalize' 的属性或方法
        if train_opt.get('generalize_first') and hasattr(self.unet, 'generalize'):
            self.unet.generalize()

        self.unet.finetune(aux=self.oomn_iter > 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        align_params, oomn_params = [], []

        for k, v in self.ddpm.named_parameters():
            if v.requires_grad:
                if '.align_weights' in k or '.align_biases' in k:
                    align_params.append(v)
                elif '.aux_weight' in k or '.aux_bias' in k:
                    oomn_params.append(v)
                else:
                    logger.warning(f'Params {k} will not be optimized, though it requires grad!')
            else:
                logger.warning(f'Params {k} will not be optimized.')

        optim_align_type = train_opt['align_opt']['optim_g'].pop('type')
        self.optimizer_align = self.get_optimizer(optim_align_type, align_params, **train_opt['align_opt']['optim_g'])
        self.optimizers.append(self.optimizer_align)
        self.cur_optimizer = self.optimizer_align

        if self.oomn_iter > 0:
            optim_oomn_type = train_opt['oomn_opt']['optim_g'].pop('type')
            self.optimizer_oomn = self.get_optimizer(optim_oomn_type, oomn_params, **train_opt['oomn_opt']['optim_g'])
            self.optimizers.append(self.optimizer_oomn)


    def setup_schedulers(self):
        def get_scheduler_class(scheduler_type):
            if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
                return lr_scheduler.MultiStepRestartLR
            elif scheduler_type == 'CosineAnnealingRestartLR':
                return lr_scheduler.CosineAnnealingRestartLR
            else:
                raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

        train_opt = self.opt['train']
        optim_align_type = train_opt['align_opt']['scheduler'].pop('type')
        self.scheduler_align = get_scheduler_class(optim_align_type)(self.optimizer_align,
                                                                     **train_opt['align_opt']['scheduler'])
        self.schedulers.append(self.scheduler_align)
        self.cur_scheduler = self.scheduler_align

        if self.oomn_iter > 0:
            optim_oomn_type = train_opt['oomn_opt']['scheduler'].pop('type')
            self.scheduler_oomn = get_scheduler_class(optim_oomn_type)(self.optimizer_oomn,
                                                                       **train_opt['oomn_opt']['scheduler'])
            self.schedulers.append(self.scheduler_oomn)

    def feed_data(self, data, phase=None):
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
        if current_iter == (self.align_iter + 1):
            logger = get_root_logger()
            logger.info('Switch to optimize oomn branch....')
            self.cur_optimizer = self.optimizer_oomn
            self.cur_scheduler = self.scheduler_oomn
        self.cur_optimizer.zero_grad()
        pred_noise, noise, x_recon_cs, x_start, t, color_scale = self.ddpm(self.HR, self.LR,
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
        self.cur_optimizer.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def save(self, epoch, current_iter, label=None):
        denoise_fn_bare_module, color_fn = self.ddpm.deploy()
        if isinstance(self.unet, RepNRBase):
            self.save_network([denoise_fn_bare_module, color_fn], 'net_g', current_iter, label, param_key=['params_deploy_denoise_fn', 'params_color_fn'])
        else:
            self.save_network([self.ddpm], 'net_g', current_iter, label, param_key=['params'])


