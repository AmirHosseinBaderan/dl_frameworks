# Step 3 — Autograd (Automatic Differentiation)

## Goal
Understand how PyTorch computes gradients automatically — the engine behind training
neural networks. `autograd` records every operation on tensors that have
`requires_grad=True` and builds a computation graph so it can differentiate via the
chain rule.

## How it works
- Create a tensor with `requires_grad=True` (e.g. `x = torch.tensor(2.0, requires_grad=True)`).
- Build a computation (e.g. `y = 3 * x + 5`).
- Call `y.backward()` to compute `dy/dx` and store it in `x.grad`.
- The chain rule works for nested expressions (`a = x + 1`, `b = a**2`, `c = b * 3`).
- For non-scalar outputs, reduce to a scalar first (e.g. `loss = y2.mean()`) before
  calling `backward()`.
- Example: with `w` as the learnable weight and a squared-error loss, `loss.backward()`
  gives the gradient used to update `w`.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Demonstrates `requires_grad`, `backward()`, the chain rule, vector gradients, and a simple weight-gradient example. |
