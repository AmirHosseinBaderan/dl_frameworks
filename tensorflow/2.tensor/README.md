# Step 2 — Tensors & Operations

## Goal
Learn TensorFlow's core data structure, the **`tf.Tensor`**, and the operations you can
perform on it. This mirrors PyTorch's `2.tensor` step.

## How it works
- **Creation**: `tf.constant(...)` for scalars, vectors, matrices, and 3D tensors.
- **Properties**: `.shape`, `tf.rank(t)`, `tf.size(t)`, `.dtype`.
- **Conversion**: `tf.constant(np_array)` (NumPy → Tensor) and `tensor.numpy()`
  (Tensor → NumPy).
- **Operations** (see [`operations.py`](operations.py)): element-wise `+ - * /`,
  `tf.nn.relu`, reductions (`tf.reduce_sum/mean/min/max` with `axis`), broadcasting,
  `tf.matmul`, slicing/indexing, `tf.cast` (type casting), and `tf.reshape`/`flatten`.

> Broadcasting rule: if shapes align for the operation, TensorFlow stretches them
> automatically (e.g. a `(2,3)` matrix + a `(3,)` vector works).

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Creates tensors of various ranks, prints their properties, and converts to/from NumPy. |
| [`operations.py`](operations.py) | Arithmetic, reduction, broadcasting, matrix, slicing, casting, and reshape operations. |
