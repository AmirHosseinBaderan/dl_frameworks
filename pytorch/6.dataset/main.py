import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.randn(1000, 5)
        self.y = torch.randn(1000, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = MyDataset()
for i in range(10):
    print(dataset[i])
