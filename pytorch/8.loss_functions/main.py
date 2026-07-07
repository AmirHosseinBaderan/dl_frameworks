"""
What is a loss function?

In a nutshell:

A function that measures the distance between the model's prediction and the actual answer.
"""

import torch
import torch.nn as nn

criterion = nn.MSELoss()

prediction = torch.tensor(
    [2., 4., 6.]
)

target = torch.tensor(
    [1., 5., 7.]
)

loss = criterion(prediction, target)
print(loss)

# Binary Classification
criterion = nn.BCELoss() # BCEWithLogitsLoss -> SigmoId + BCELoss

# criterion = nn.CrossEntropyLoss() -> use for multi-class classification (Like MNIST) , its do Softmax

prediction = torch.tensor(
    [0.8,0.2,0.9]
)

target = torch.tensor(
    [1.,0.,1.]
)

loss = criterion(
    prediction,
    target
)
print(loss)

"""
Which loss function should be used, and when?

| مسئله                           | Loss                     |
| ------------------------------- | ------------------------ |
| Regression                      | `nn.MSELoss()`           |
| Regression More resistant to outliers| `nn.HuberLoss()`         |
| Binary Classification           | `nn.BCEWithLogitsLoss()` |
| Multi-Class Classification      | `nn.CrossEntropyLoss()`  |

"""

"""
| Issue                | Correct answer      |
| ---------------------| ------------------- |
| House prices         |  MSE یا Huber       |
| Spam Detection       |  BCEWithLogitsLoss  |
| MNIST                |  CrossEntropyLoss   |
| Temperature forecast |  MSE or Huber |
"""

# for multi class -> nn.CrossEntropyLoss() , for multi label -> nn.BCEWithLogitsLoss()

"""
| Problem Type               | Output Format                  | Loss                    |
| -------------------------- | ------------------------------ | ----------------------- |
| Regression                 | A numerical value              | `MSELoss` / `HuberLoss` |
| Binary Classification      | One of two states              | `BCEWithLogitsLoss`     |
| Multi-Class Classification | Exactly one class**            | `CrossEntropyLoss`      |
| Multi-Label Classification | Zero, one, or multiple classes simultaneously** | `BCEWithLogitsLoss`     |
"""