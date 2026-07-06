import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        print(x.shape)

        x = self.fc1(x)
        print(x.shape)

        x = torch.relu(x)
        print(x.shape)

        x = self.fc2(x)
        print(x.shape)

        return x


model = MyModel()
x = torch.randn(5,4)
y = model(x)
print(y)
print(y.shape)

# forward is only prediction
# so where is backward?
# after forward :
"""
prediction = model(x)

loss = criterion(
    prediction,
    target
)

loss.backward()
"""
