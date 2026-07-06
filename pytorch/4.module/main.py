import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.linear = nn.Linear(3, 2)

        # multi layer
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
print(model)

x = torch.randn(32,10 )
print(x)
print(x.shape)
y = model(x)
print(y.shape)

# model parameters
for p in model.parameters():
    print(p.shape)

# parameters count
total = sum(
    p.numel()
    for p in model.parameters()
)
print(total)

model.train()

# move to gpu
# model.to("cuda")