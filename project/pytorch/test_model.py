import torch
from model import FashionClassifier

model = FashionClassifier()
x = torch.randn(64,1,28,28)

output = model(x)
print(output.shape)