# Step 13 — Saving & Loading Models

## Goal
Persist trained models and resume training via checkpoints. This is the final PyTorch
step and is essential for deployment and long training runs.

## How it works
- **Save weights only**: `torch.save(model.state_dict(), "model.pth")`.
- **Load weights**: rebuild the architecture first, then
  `model.load_state_dict(torch.load("model.pth"))`. The `.pth` file stores only
  parameters, so the model class must be re-created.
- After loading, call `model.eval()` (Dropout/BatchNorm behave differently in train vs
  test) and wrap inference in `torch.no_grad()`.
- **Checkpoint** (resume training): save a dict with `epoch`, `model`, and `optimizer`
  state dicts; restore all three to continue training.
- Flow: Train → best model → `torch.save()` → `model.pth` → `torch.load()` → predict.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Documents (via comments) saving `state_dict`, loading, `model.eval()`, `torch.no_grad()`, and full checkpoints for resuming training. |
