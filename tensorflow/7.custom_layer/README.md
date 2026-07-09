# Step 7 — Custom Layers

## Goal
Create your own Keras layers by subclassing `tf.keras.Layer`. This is how you extend
Keras beyond the built-in layers — the TensorFlow counterpart to defining custom
modules/blocks in PyTorch.

## How it works
- Subclass `tf.keras.Layer` and implement `call(self, inputs)` (the forward logic).
- Use `self.add_weight(...)` to register trainable parameters; `build(input_shape)` is
  called lazily once the input size is known, so you can create shape-dependent weights.
- Custom layers drop into `Sequential` or the Functional API like any built-in layer.
- Examples: `DoubleLayer` (×2), `MyLayer` (learned scalar weight), `MyDense` /
  `MyDenseWithActivation` (manual `matmul` + bias + ReLU), and `MyLinear(units)`.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Instantiates the custom layers, uses them inside `Sequential`/`Functional` models, and prints trainable variables. |
| [`custom_layers.py`](custom_layers.py) | Defines `DoubleLayer`, `MyLayer`, `MyDense`, `MyDenseWithActivation`, `AddTen`, `Square`, and `MyLinear`. |
