# Step 9 — Optimizers

## Goal
Learn how optimizers update model weights using the gradients computed by
backpropagation. In PyTorch, optimizers live in `torch.optim`.

## How it works
- Create an optimizer over `model.parameters()`:
  - `optim.SGD(model.parameters(), lr=0.01)`
  - `optim.Adam(model.parameters(), lr=0.001)`
  - `optim.RMSprop(...)`
- The **correct training order** is critical:
  1. `optimizer.zero_grad()` — clear old gradients (otherwise they accumulate).
  2. `prediction = model(x)`
  3. `loss = criterion(prediction, y)`
  4. `loss.backward()`
  5. `optimizer.step()` — apply the update.
- Variants: SGD with `momentum=0.9`, Adam with `weight_decay` (L2 regularization).

| Problem           | Optimizer      |
| ----------------- | -------------- |
| Most DL projects  | Adam           |
| Classic CNNs      | SGD + Momentum |
| Transformers      | AdamW          |
| Research          | Adam           |

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Defines a model, shows SGD/Adam/RMSprop setup, the mandatory zero_grad→backward→step order, momentum, and weight decay. |
