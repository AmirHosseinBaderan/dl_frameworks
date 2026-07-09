# Step 4 — `GradientTape` (Automatic Differentiation)

## Goal
Learn TensorFlow's automatic differentiation tool, `tf.GradientTape` — the equivalent
of PyTorch's `autograd`/`loss.backward()`. It records operations so gradients can be
computed afterward.

## How it works
- Wrap the forward computation in `with tf.GradientTape() as tape:`.
- `tape.gradient(y, x)` returns `dy/dx`.
- Works for multiple variables: `tape.gradient(z, [x, y])`.
- By default, `GradientTape` only tracks `tf.Variable`s. To track a constant, call
  `tape.watch(x)` first.
- Full update example: compute `loss = (w - 5)**2`, get `grad`, then
  `w.assign_sub(learning_rate * grad)` — this is one manual gradient-descent step.

This is the engine behind the custom training loop in Step 8.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Shows `GradientTape` gradients for single/multi variables, `watch` on constants, and a manual weight update. |
