# Step 9 — `tf.data` Pipeline

## Goal
Build efficient data input pipelines with `tf.data.Dataset` — TensorFlow's equivalent of
PyTorch's `Dataset` + `DataLoader`, but with built-in shuffling, mapping, batching,
caching, and prefetching.

## How it works
- Create from arrays: `tf.data.Dataset.from_tensor_slices(x)`.
- Chain transformations:
  - `.shuffle(buffer_size)` — randomize.
  - `.map(fn)` — apply a function per element (e.g. normalize, square).
  - `.batch(n)` — group into batches.
  - `.cache()` — keep data in memory (good for static sets like MNIST).
  - `.prefetch(tf.data.AUTOTUNE)` — prepare the next batch on CPU while the GPU trains.
  - `.repeat()` — loop forever.
- The standard order is: `from_tensor_slices → cache → shuffle → map → batch → prefetch`.
- Inside a training loop, iterate `for batch_x, batch_y in dataset:` and use the
  `GradientTape` pattern.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Demonstrates `from_tensor_slices`, shuffle/map/batch, caching, prefetch, and a standard pipeline composition. |
