# Step 10 — Callbacks

## Goal
Use Keras **callbacks** to run custom code at specific points during `model.fit()`
(e.g. save the best model, stop early, log to CSV/TensorBoard). This is a high-level
convenience PyTorch does not have built in.

## How it works
Common callbacks:
- `ModelCheckpoint` — save the best model (`save_best_only`, `monitor='val_loss'`).
- `EarlyStopping` — stop when no improvement for `patience` epochs
  (`restore_best_weights=True`).
- `ReduceLROnPlateau` — lower the learning rate when progress stalls.
- `CSVLogger` — append loss/metric history to a CSV.
- `TensorBoard` — write logs for the TensorBoard UI.
- Custom callback: subclass `tf.keras.callbacks.Callback` and override hooks like
  `on_epoch_end`.

The "best combination" block wires all of these into a single `callbacks` list passed to
`model.fit(..., callbacks=[...])`.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Sets up `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `CSVLogger`, `TensorBoard`, and a recommended combination. |
| [`custom.py`](custom.py) | `MyCallback` — a custom callback printing metrics each epoch and stopping at 99% accuracy. |
| [`sample.py`](sample.py) | A full runnable example: builds a model, compiles, and trains with the callbacks for 10 epochs. |
| `best.keras` | The best saved model artifact produced by `ModelCheckpoint`. |
| `logs/` | TensorBoard event files written during training. |
