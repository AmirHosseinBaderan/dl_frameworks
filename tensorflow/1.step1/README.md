# Step 1 — Setup & Verify Installation

## Goal
Confirm TensorFlow is installed and check which devices (CPU/GPU) are available.

## How it works
- `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"` silences TensorFlow's verbose INFO logs.
- `tf.__version__` prints the installed version.
- `tf.config.list_physical_devices("GPU")` lists available GPUs.
- `tf.config.list_physical_devices()` lists all physical devices.

This is the TensorFlow equivalent of PyTorch's `1.step1` environment check.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Imports TensorFlow, prints version and detected CPU/GPU devices. |
