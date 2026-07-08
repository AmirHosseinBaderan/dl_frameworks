
"""
save model
torch.save(
    model.state_dict(),
    "model.pth"
)
"""

"""
load model 
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)
first make model architecture again
then 
model.load_state_dict(
    torch.load("model.pth")
)

Why do we build the model first?
Because the `model.pth` file only stores the parameters.

after load 
model.eval()

Why?
Because layers such as:
.-.Dropout
.-.BatchNorm
behave differently during training and testing.

for prediction 
with torch.no_grad():
    prediction = model(x)
    
Always use `torch.no_grad()` for inference.
"""

"""
Saving a Checkpoint

Sometimes, the model alone isn't enough.
You want to be able to resume training if it gets interrupted.
In that case, we save the optimizer and the epoch alongside the model.

torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict()
}, "checkpoint.pth")

load 

checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(
    checkpoint["model"]
)

optimizer.load_state_dict(
    checkpoint["optimizer"]
)

epoch = checkpoint["epoch"]

"""

"""
Checkpoint vs Model

Model :
Weights
↓
Prediction

It is suitable for deployment.

Checkpoint : 
Weights
Optimizer
Epoch
↓
Resume Training

It is suitable for continuing the training.
"""

"""
Train
      │
      ▼
Best Model
      │
      ▼
torch.save()
      │
      ▼
model.pth
      │
      ▼
torch.load()
      │
      ▼
Prediction
"""