import torch.utils.data as data


class ELDTrainDataset(data.Dataset):
    def __init__(self, target_dataset, input_datasets, size=None, flag=None, augment=True, cfa='bayer'):
        super(ELDTrainDataset, self).__init__()
        self.size = size
        self.target_dataset = target_dataset
        self.input_datasets = input_datasets
        self.flag = flag
        self.augment = augment
        self.cfa = cfa

    def __getitem__(self, i):
        N = len(self.target_dataset)
        target_image = self.target_dataset[i // N]

        if self.input_datasets is None:
            input_image = self.target_dataset[i // N]
        else:
            input_image = self.input_datasets[i % N][i // N]

        return target_image, input_image

    def __len__(self):
        size = self.size or len(self.target_dataset) * len(self.input_datasets)
        return size
