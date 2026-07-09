"""
        nn.Module
              │
     ┌────────┴────────┐
     │                 │
 __init__()        forward()
     │                 │
     │                 │
Layer definitions  Data flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # use start_dim to not change batch -> x is (64,1,28,28) , after start_dim its be (1,28,28)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x
