# Step 10 — Custom Training Loop

## Goal
Put everything together (Dataset → DataLoader → Model → Loss → Optimizer) into an
explicit training loop — the heart of PyTorch's "you write the loop" philosophy.

## How it works
The data flow for each batch is:

```
Dataset → DataLoader → Batch (x, y)
        → prediction = model(x)
        → loss = criterion(prediction, y)
        → loss.backward()
        → optimizer.step()
        → optimizer.zero_grad()  → next batch
```

- `MyDataset` provides 100 random samples of 4 features with labels in `{0,1,2}`.
- A `DataLoader` batches them (`batch_size=16`, `shuffle=True`).
- The model is a small `nn.Sequential` (Linear 4→8, ReLU, Linear 8→3).
- `CrossEntropyLoss` + `Adam(lr=0.001)` train for 5 epochs, printing the running loss.

> Note: there is no separate `model.py` in this folder; the model is defined inline with
> `nn.Sequential` inside [`main.py`](main.py).

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | The full training loop: builds dataset, loader, model, loss, optimizer, and runs 5 epochs. |
| [`dataset.py`](dataset.py) | `MyDataset(Dataset)` generating random `(x, y)` pairs for the loop. |
