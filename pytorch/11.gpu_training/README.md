# Step 11 — GPU Training

## Goal
Learn how to move models and data to a GPU (CUDA) for faster training, and the key
rule that **all tensors in an operation must be on the same device**.

## How it works
- Pick a device: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Move the model: `model = model.to(device)`.
- Move each batch: `x = x.to(device)`, `y = y.to(device)` inside the loop.
- For brand-new tensors, specify the device directly: `torch.zeros(10, device=device)`.
- CPU path: RAM → CPU → training. GPU path: RAM → VRAM → training.

The file is mostly annotated (commented) to explain the pattern without running a full
training job, but the commented training loop shows exactly where `.to(device)` belongs.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Explains CPU vs GPU data flow and shows (commented) how to move model and batches to the GPU. |
