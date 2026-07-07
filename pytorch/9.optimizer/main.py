import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

model = MyModel()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

# correct order
"""
optimizer.zero_grad()

prediction = model(x)

loss = criterion(prediction,y)

loss.backward()

optimizer.step()
"""

# If you forget this: optimizer.zero_grad()
# The model exhibits strange behavior.
# This is because the gradients accumulate.

# adam
"""
optimizer = optim.Adam(

    model.parameters(),

    lr=0.001
)
"""

# RMSProp
# optim.RMSprop(...)

# SGD
# optim.SGD(...)

# Momentum
"""
optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
"""

# Weight Decay , L2 Regularization
"""
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)
"""

# The best choice?
"""
| Problem           | Optimizer      |
| ----------------- | -------------- |
| Most DL projects  | Adam           |
| Classic CNNs      | SGD + Momentum |
| Transformers      | AdamW          |
| Research          | Adam           |
"""