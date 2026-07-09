# Step 8 — Loss Functions

## Goal
Understand loss (criterion) functions — they measure the distance between the model's
prediction and the true target, and their value is what backpropagation minimizes.

## How it works
PyTorch provides losses under `torch.nn`:
- `nn.MSELoss()` — regression.
- `nn.BCEWithLogitsLoss()` — binary classification (sigmoid + BCE in one, numerically stable).
- `nn.CrossEntropyLoss()` — multi-class classification (applies softmax internally).
- `nn.HuberLoss()` — regression robust to outliers.

The script computes a sample MSE and a BCE loss, then documents which loss to use for
each problem type via tables (regression, binary, multi-class, multi-label).

| Problem Type               | Output Format                  | Loss                    |
| -------------------------- | ------------------------------ | ----------------------- |
| Regression                 | A numerical value              | `MSELoss` / `HuberLoss` |
| Binary Classification      | One of two states              | `BCEWithLogitsLoss`     |
| Multi-Class Classification | Exactly one class              | `CrossEntropyLoss`      |
| Multi-Label Classification | Zero, one, or multiple classes | `BCEWithLogitsLoss`     |

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Demonstrates `MSELoss`/`BCELoss` and documents loss selection with reference tables. |
