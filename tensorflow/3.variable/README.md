# Step 3 — `tf.Variable`

## Goal
Understand `tf.Variable`, the wrapper TensorFlow uses for **learnable parameters**
(weights, biases, embeddings, convolution kernels). This is the TensorFlow counterpart
to PyTorch's `nn.Parameter`.

## How it works
- Create with `tf.Variable([1, 2, 3])` or by wrapping a tensor/NumPy array.
- Update in place with `w.assign([...])`, `w.assign_add([...])`, `w.assign_sub([...])`.
- Variables behave like constants in math ops (`w * 2`, `tf.reduce_sum(w)`).
- **What is a Variable?** Almost all learnable parameters.
- **What is NOT a Variable?** Inputs, labels, datasets, and predictions.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Demonstrates creating, assigning, and updating variables, plus conversion to/from tensors and NumPy. |
