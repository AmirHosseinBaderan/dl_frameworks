# Step 1 — Setup & Verify Installation

## Goal
Confirm that PyTorch is installed correctly and check whether a CUDA-capable GPU is
available in the current environment.

## How it works
The script simply imports `torch` and prints two pieces of information:
- `torch.__version__` — the installed PyTorch version.
- `torch.cuda.is_available()` — `True` if PyTorch can see a GPU, otherwise `False`.

This is the "hello world" of any PyTorch environment check before running real code.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Imports `torch` and prints the version and GPU availability. |
