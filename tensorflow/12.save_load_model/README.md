# Step 12 — Saving & Loading Models

## Goal
Persist and reload trained models. TensorFlow's `model.save()` / `load_model()` is
higher-level than PyTorch's `state_dict` approach — it saves the full model (architecture
+ weights + optimizer state) in one file.

## How it works
- **Full model**: `model.save("my_model.keras")` then
  `tf.keras.models.load_model("my_model.keras")`.
- **Weights only**: `model.save_weights("weights.weights.h5")` /
  `model.load_weights("weights.weights.h5")` — useful when the architecture already lives
  in code/Git.
- `ModelCheckpoint` (Step 10) uses `model.save` under the hood.
- When loading a model that uses **custom layers/losses**, pass
  `custom_objects={"MyDense": MyDense}` so Keras knows how to rebuild them.

The script trains a small model, saves it as `demo.keras`, reloads it, and predicts.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Trains a model, saves it with `model.save`, reloads with `load_model`, and predicts; documents weights-only and custom-object loading. |
| `demo.keras` | The saved full model artifact produced by the script. |
