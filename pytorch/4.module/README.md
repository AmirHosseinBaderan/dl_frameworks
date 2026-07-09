# Step 4 — `nn.Module` (Build a Model)

## Goal
Learn to define a neural network by subclassing `torch.nn.Module`, the base class for
all PyTorch models. You declare layers in `__init__` and define the data flow in
`forward`.

## How it works
- Subclass `nn.Module` and call `super().__init__()`.
- In `__init__`, create layers such as `nn.Linear(in, out)` (fully connected).
- In `forward(self, x)`, chain the layers with activations like `F.relu`.
- Instantiate the model and call it like a function: `y = model(x)`.
- Inspect parameters with `model.parameters()` and count them with `p.numel()`.
- `model.train()` switches to training mode; `model.to("cuda")` moves it to GPU.

This is the standard PyTorch pattern you will reuse in every later step and in the
end-to-end project.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Defines `MyModel(nn.Module)` with three linear layers, runs a forward pass, and prints parameter shapes/counts. |
