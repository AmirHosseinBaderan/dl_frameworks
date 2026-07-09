# Step 8 — Custom Training Loop

## Goal
Write a training loop by hand using `GradientTape`, instead of `model.fit()`. This is
the TensorFlow counterpart to PyTorch's explicit loop and gives you full control.

## How it works
When you call `model.fit(x, y)`, TensorFlow roughly does this per batch:
`Forward → Loss → Backward → Update Weights`. The script reproduces that manually:

```python
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        prediction = model(x)
        loss = loss_fn(y, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

- `loss_fn = tf.keras.losses.MeanSquaredError()`
- `optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)`
- `tape.gradient(loss, model.trainable_variables)` computes gradients for all weights.
- `optimizer.apply_gradients(...)` applies the update.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Builds a small model and trains it for 100 epochs with a manual `GradientTape` loop, printing the loss each epoch. |
