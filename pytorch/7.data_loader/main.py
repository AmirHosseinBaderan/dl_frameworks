import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.randn(1000, 5)
        self.y = torch.randn(1000, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = MyDataset()

# each batch most be 32 items if one dataset after split batches be 32,32,32....3 then we can use drop_last=True to remove it
# shuffle -> For each epoch, the data order should be randomized., boolean
# num_workers -> count of threads , number
# pin_memory -> move RAM to GPU ,boolean
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=0
)

for x, y in loader:
    print("X Shape : ", x.shape)
    print("Y Shape : ", y.shape)
    break
