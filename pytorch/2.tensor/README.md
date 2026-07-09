# Step 2 — Tensors & Operations

## Goal
Learn the fundamental data structure of PyTorch — the **tensor** — and the common
operations you perform on it. A tensor is a multi-dimensional array (scalar, vector,
matrix, or higher rank) that can live on CPU or GPU.

## How it works
- **Creation**: `torch.tensor(...)`, `torch.zeros`, `torch.ones`, `torch.full`,
  `torch.rand`, `torch.randn`, `torch.arange`, `torch.linspace`.
- **Properties**: `.shape`, `.ndim`, `.numel()`, `.size()`, `.dtype`, `.device`.
- **Conversion**: `torch.from_numpy(a)` and `x.numpy()` to move between NumPy and PyTorch.
- **Shape manipulation**: `reshape`, `view`, `unsqueeze`, slicing (`x[2:6]`, `x[:, 2]`).
- **Operations** (see [`operations.py`](operations.py)): element-wise `+ - * /`,
  `pow`, `sqrt`, `log`, `exp`, `sin/cos/tan`, reductions (`mean`, `sum`, `max`,
  `argmax`), matrix multiplication (`@` / `torch.matmul`), `transpose`, `flatten`,
  `cat`/`stack`, broadcasting, boolean masking, and in-place ops (`add_`).

> Note: `view()` only works on contiguous tensors; `reshape()` is safer because it
> copies when needed.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Tensor creation, properties, NumPy conversion, reshape/view/unsqueeze, and slicing. |
| [`operations.py`](operations.py) | Arithmetic, math, reduction, matrix, and shape operations with examples. |
