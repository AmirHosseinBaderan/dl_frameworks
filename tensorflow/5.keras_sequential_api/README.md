# Step 5 — Keras Sequential API

## Goal
Build a neural network with Keras' high-level `tf.keras.Sequential` API — the easiest
way to stack layers linearly. This is TensorFlow's equivalent of a simple `nn.Sequential`.

## How it works
- Stack layers: `tf.keras.Sequential([Dense(128, activation="relu"), Dense(64, ...), Dense(10, activation="softmax")])`.
- **Build** the model so it knows input shape: `model.build((None, 784))` or pass an
  `Input` layer / run a dummy forward pass.
- `model.summary()` prints the layer shapes and parameter counts.
- Forward pass: `y = model(x)` (e.g. random `(5, 784)` input → `(5, 10)` output).
- Layers can also be added incrementally with `model.add(...)`.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Builds Sequential models, demonstrates `build`/`Input`/`add`, prints summaries, and runs a sample forward pass. |
