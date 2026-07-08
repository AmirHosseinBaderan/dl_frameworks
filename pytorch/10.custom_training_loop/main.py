"""
          Dataset
             │
             ▼
        DataLoader
             │
             ▼
        Batch (x, y)
             │
             ▼
     prediction = model(x)
             │
             ▼
 loss = criterion(prediction, y)
             │
             ▼
     loss.backward()
             │
             ▼
     optimizer.step()
             │
             ▼
     optimizer.zero_grad()
             │
             └──────────────► Batch بعدی
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyDataset

dataset = MyDataset()

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# training loop
for epoch in range(5):
    running_loss = 0

    for x, y in loader:
        optimizer.zero_grad()

        prediction = model(x)

        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch}, Loss {loss.item():.4f}")
    print("Loss Mean : ", running_loss / len(loader))
