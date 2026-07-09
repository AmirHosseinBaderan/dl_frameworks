# Step 5 — Forward Pass

## Goal
See what happens during a forward pass: data flows through the layers and the shape
changes at each step. The forward pass only produces predictions — backpropagation
happens separately via `loss.backward()`.

## How it works
- Define `MyModel` with `nn.Linear(4, 8)` and `nn.Linear(8, 3)`.
- Print `x.shape` at each stage: input `(5, 4)` → after `fc1` `(5, 8)` → after `relu`
  unchanged → after `fc2` `(5, 3)`.
- `model(x)` triggers `forward` and returns the output.
- The commented block shows the full training pattern: `prediction = model(x)`,
  `loss = criterion(prediction, target)`, then `loss.backward()`.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | A small model that prints tensor shapes at every layer to visualize the forward pass. |
