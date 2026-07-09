# Step 6 — `Dataset`

## Goal
Learn to create a custom dataset by subclassing `torch.utils.data.Dataset`. A dataset
tells PyTorch how to load a single sample `(x, y)` and how many samples exist.

## How it works
- Subclass `Dataset` and implement three things:
  - `__init__`: prepare/store the data (here random `x` of shape `(1000, 5)` and `y`
    of shape `(1000, 1)`).
  - `__len__`: return the number of samples (`len(self.x)`).
  - `__getitem__(idx)`: return the sample at `idx` as `(x[idx], y[idx])`.
- Indexing the dataset (`dataset[i]`) returns one `(input, target)` pair.

This is the foundation for the `DataLoader` in the next step.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Defines `MyDataset(Dataset)` with random data and prints the first 10 samples. |
