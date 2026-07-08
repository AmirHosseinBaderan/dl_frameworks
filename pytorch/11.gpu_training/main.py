"""
CPU

RAM
↓
CPU
↓
Training

"""
import torch

"""
GPU

RAM
↓
GPU Memory (VRAM)
↓
Training
"""

# The most important PyTorch rule:
# All tensors involved in an operation must be on the same device.
# For example, this is incorrect:
# Wrong Way
"""
Model → GPU
Input → CPU
"""

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)

# move model
# model = MyModel().to(device)
# move data
# x = x.to(device) , y = y.to(device)
"""
for x, y in loader:

    x = x.to(device)
    y = y.to(device)

    ...
"""

"""
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

for epoch in range(epochs):

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        prediction = model(x)

        loss = criterion(prediction, y)

        loss.backward()

        optimizer.step()
"""

# for new tensor we use
# z = torch.zeros(10, device=device)