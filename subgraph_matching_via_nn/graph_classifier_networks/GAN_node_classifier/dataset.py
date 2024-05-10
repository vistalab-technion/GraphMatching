import torch
from torch.utils.data import Dataset


class gan_dataset(Dataset):
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