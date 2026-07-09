# Step 7 — `DataLoader`

## Goal
Wrap a `Dataset` in a `DataLoader` to get batched, shuffled, and (optionally)
multi-process data loading — exactly what the training loop consumes.

## How it works
- `DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)`.
  - `batch_size`: number of samples per batch.
  - `shuffle`: randomize order each epoch.
  - `drop_last`: drop the final smaller batch if `True`.
  - `num_workers`: number of subprocesses for loading.
  - `pin_memory`: speed up host→GPU transfer.
- Iterating the loader yields batches `(x, y)` of shape `(32, 5)` and `(32, 1)`.

This builds directly on the `Dataset` from Step 6.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Creates `MyDataset`, wraps it in a `DataLoader`, and prints the shape of the first batch. |
