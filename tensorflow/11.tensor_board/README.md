# Step 11 — TensorBoard

## Goal
Use **TensorBoard** to visualize training metrics, histograms, and images. TensorBoard
is TensorFlow's built-in visualization suite (PyTorch typically uses separate tools
like Weights & Biases or TensorBoardX).

## How it works
- Launch: `tensorboard --logdir ./logs` then open `http://localhost:6006`.
- Write manual logs with a file writer:
  ```python
  writer = tf.summary.create_file_writer('./logs/custom')
  with writer.as_default():
      tf.summary.scalar("loss", 0.52, step=1)
  ```
- Other summaries: `tf.summary.histogram("weights", ...)` and `tf.summary.image(...)`.
- During `model.fit`, the `TensorBoard` callback (Step 10) writes logs automatically.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Shows how to create a file writer and log a scalar manually; documents histogram/image logging. |
