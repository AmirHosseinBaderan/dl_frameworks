# Step 12 — TorchVision & Datasets

## Goal
Use `torchvision` to download a standard dataset (MNIST), wrap it in a `DataLoader`,
and apply image transforms (resize, to-tensor, normalize).

## How it works
- `transforms.ToTensor()` converts PIL images to tensors.
- `datasets.MNIST(root="./data", train=True, download=True, transform=transform)`
  downloads and prepares the data automatically.
- A `DataLoader` batches it (`batch_size=32`, `shuffle=True`); `next(iter(loader))`
  returns a batch of `(images, labels)`.
- `transforms.Resize`, `transforms.Normalize`, and `transforms.Compose` build a
  transform pipeline (resize → tensor → normalize to `[-1, 1]`).
- `matplotlib` visualizes a sample image with its label.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Loads MNIST via torchvision, builds a DataLoader, and demonstrates transforms and image display. |
