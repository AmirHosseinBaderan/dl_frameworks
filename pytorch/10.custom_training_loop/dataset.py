from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):

    def __init__(self):
        self.x = torch.randn(100, 4)
        self.y = torch.randint(0, 3, (100,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

