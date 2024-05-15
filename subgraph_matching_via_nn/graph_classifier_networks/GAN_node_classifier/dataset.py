import torch
from bgan_pytorch.bgan.datasets import BaseQuantizedImageDataset

class gan_dataset(BaseQuantizedImageDataset):
    def __init__(self, dataframe):
        self.data = dataframe.values.astype(float)
        self.labels = dataframe.values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'input': torch.tensor(self.data[idx]),
            'label': torch.tensor(self.labels[idx])
        }
        return sample

    @property
    def num_colors(self):
        return 2 # as the distribution is binary per feature

    # def dequantize(self, img):
    #     return img