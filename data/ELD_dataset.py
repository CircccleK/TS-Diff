import os

import torch.utils.data as data

from data.ll_dataset import LL_Dataset
from utils.raw_data import read_data, read_paired_fns
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ELDDataset(data.Dataset):
    def __init__(self, opt):
        super(ELDDataset, self).__init__()
        self.opt = opt

        self.basedir = opt['databasedir'] + opt['camera']
        self.fns = read_paired_fns(opt['fns_root'])
        self.input_paths = [os.path.join(self.basedir, fn[0]) for fn in self.fns]
        self.gt_paths = [os.path.join(self.basedir, fn[1]) for fn in self.fns]
        self.ratios = [float(fn[2]) for fn in self.fns]

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        input_path = self.input_paths[index]
        gt_raw = read_data(gt_path)
        input_raw = read_data(input_path)

        input_raw = input_raw * self.ratios[index]

        position_encoding = LL_Dataset.get_position_encoding(input_raw, self.opt)

        input_raw, gt_raw, position_encoding = LL_Dataset.crop_raw(input_raw, gt_raw, position_encoding, self.opt)

        return LL_Dataset.process(input_raw, input_path, gt_raw, gt_path, position_encoding, self.opt)

    def __len__(self):
        return len(self.gt_paths)