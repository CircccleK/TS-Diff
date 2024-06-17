import os

import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from utils import noise
from utils.data_utils import hiseq_color_cv2_img, generate_position_encoding
from utils.raw_data import read_data, read_paired_fns, \
    postprocess_bayer, compute_expo_ratio, raw2rgb_batch
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LL_Dataset(data.Dataset):

    def __init__(self, opt):
        super(LL_Dataset, self).__init__()
        self.opt = opt

        self.gt_root = opt['gt_root']
        self.input_root = opt['input_root']
        self.fns = read_paired_fns(opt['fns_root'])
        self.gt_paths = [os.path.join(opt['gt_root'], fn[1]) for fn in self.fns]
        self.input_paths = [os.path.join(opt['input_root'], fn[0]) for fn in self.fns]
        self.mean = self.opt['mean']
        self.std = self.opt['std']
        # if self.opt['name'] == 'train':
        #     target_data = lmdb_dataset.LMDBDataset(opt['db_gt_dir'])
        #     input_data = lmdb_dataset.LMDBDataset(opt['db_input_dir']) if opt['db_input_dir'] is not None else None
        #     self.train_dataset = ELDTrainDataset(target_dataset=target_data, input_datasets=input_data)

        if self.opt['name'] != 'validation' and 'camera' in self.opt and self.opt['camera'] != "Virtual":
            self.noise_model = noise.NoiseModel(self.opt['camera'])

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        input_path = self.input_paths[index]
        gt_raw = read_data(gt_path)
        input_raw = read_data(input_path)

        position_encoding = self.get_position_encoding(input_raw, self.opt)
        input_raw, gt_raw, position_encoding = self.crop_raw(input_raw, gt_raw, position_encoding, self.opt)

        if self.opt.get('read_real_data', False):
            ratio = compute_expo_ratio(input_path, gt_path, self.opt['camera'])
            input_raw = input_raw * ratio
        elif self.opt.get('read_virtual_camera', False):
            return {"LR": gt_raw, "HR": gt_raw, "lq_path": input_path, "gt_path": gt_path, "position_encoding": position_encoding}
        elif self.opt.get('read_syn_data', False):
            input_raw = self.noise_model(gt_raw)
            input_raw = np.maximum(np.minimum(input_raw, 1.0), 0)
        # elif self.opt.get('save_read_syn_data_to_rgb', False):
        #     input_raw = np.load(input_path)
        #     input_raw = np.maximum(np.minimum(input_raw, 1.0), 0)
        #     input_path = self.opt['real_data_root'] + os.path.splitext(os.path.basename(input_path))[0] + ".ARW"
        #     real_raw = read_data(input_path) * ratio
        #     lq_path = input_path
        #     return_dict = {"real_raw": real_raw, "syn_raw": input_raw, "lq_path": lq_path, "gt_path": gt_path}
        #     return return_dict
        elif self.opt.get('save_syn_data', False):
            real_raw = read_data(input_path) * ratio
            lq_path = input_path
            input_raw = self.noise_model(gt_raw)
            input_raw = np.maximum(np.minimum(input_raw, 1.0), 0)
            return_dict = {"real_raw": real_raw, "syn_raw": input_raw, "lq_path": lq_path, "gt_path": gt_path}
            return return_dict

        return self.process(input_raw, input_path, gt_raw, gt_path, position_encoding, self.opt)


    def __len__(self):
        return len(self.gt_paths)

    @staticmethod
    def process(input_raw, input_path, gt_raw, gt_path, position_encoding, opt):
        input_raw_tensor = torch.tensor(input_raw).unsqueeze(0)
        input_rgb = raw2rgb_batch([gt_path], input_raw_tensor, camera=opt['camera'])
        hiseql_img = (input_rgb.squeeze(0).cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
        del input_raw_tensor

        lq_path = input_path
        if opt['stage_in'] == 'srgb':
            input_img = postprocess_bayer(input_path, input_raw, is_Normal=True)
            gt_img = postprocess_bayer(gt_path, gt_raw, is_Normal=True)
        else:
            input_img = np.copy(input_raw).transpose((1, 2, 0))
            gt_img = np.copy(gt_raw).transpose((1, 2, 0))

        if opt.get('bright_aug', False):
            bright_aug_range = opt.get('bright_aug_range', [0.5, 1.5])
            input_img = input_img * np.random.uniform(*bright_aug_range)
            
        if opt.get('concat_with_hiseq', False):
            hiseql = hiseq_color_cv2_img(hiseql_img) / 255.
            input_img = np.concatenate([input_img, hiseql], axis=2)

        if opt.get('use_flip', False) and np.random.uniform() < 0.5:
            gt_img = np.fliplr(gt_img).copy()
            input_img = np.fliplr(input_img).copy()

        if opt.get('concat_with_position_encoding', False):
            input_img = np.concatenate([input_img, position_encoding], axis=2)

        gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
        input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))

        input_img_pt = input_img_pt.float()
        gt_img_pt = gt_img_pt.float()
        normalize(input_img_pt, [0.5] * input_img_pt.shape[0], [0.5] * input_img_pt.shape[0], inplace=True)
        normalize(gt_img_pt, [0.5] * gt_img_pt.shape[0], [0.5] * gt_img_pt.shape[0], inplace=True)

        return_dict = {"LR": input_img_pt, "HR": gt_img_pt, "lq_path": lq_path, "gt_path": gt_path}

        return return_dict

    @staticmethod
    def crop_raw(input_raw, gt_raw, position_encoding, opt):
        if opt.get('input_mode', '') == 'crop':
            crop_size = opt['crop_size']
            _, H, W = input_raw.shape
            if opt['name'] == 'train':
                h = np.random.randint(0, H - crop_size + 1)
                w = np.random.randint(0, W - crop_size + 1)
            else:
                h = H // 2 - (crop_size // 2)
                w = W // 2 - (crop_size // 2)
            gt_raw = gt_raw[:, h: h + crop_size, w: w + crop_size]
            input_raw = input_raw[:, h: h + crop_size, w: w + crop_size]
            position_encoding = position_encoding[h: h + crop_size, w: w + crop_size, :]
        elif opt.get('input_mode', '') == 'ELD':
            _, H, W = input_raw.shape
            h1 = H // 6
            h2 = H // 8
            gt_raw = gt_raw[:, h1: H - h2:, :]
            input_raw = input_raw[:, h1: H - h2, :]
            position_encoding = position_encoding[h1: H - h2, :, :]
        return input_raw, gt_raw, position_encoding

    @staticmethod
    def get_position_encoding(input_raw, opt):
        _, H, W = input_raw.shape
        L = opt.get('position_encoding_L', 1)
        position_encoding = generate_position_encoding(H, W, L)
        return position_encoding