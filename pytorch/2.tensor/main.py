import torch
import numpy as np

x = torch.tensor([1, 2, 3])
print(x)

# 2d tensor
x2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x2d)

print(x.shape)
print(x2d.shape)

print(x.ndim)
print(x2d.ndim)

# tensor element count
print(x.numel())
print(x2d.numel())

# tensor size of each dimension
print(x.size())
print(x2d.size())

print(x.dtype)
print(x2d.dtype)

# make tensor float
xf = torch.tensor(
    [1, 2, 3, 4],
    dtype=torch.float32
)
print(xf)

# device
print(x.device)

# make ready tensors
xz = torch.zeros((3, 4))
print(xz)

xo = torch.ones((3, 4))
print(xo)

# constant values
xc = torch.full((3, 4), 7)
print(xc)

# random values , between 0,1
xr = torch.rand((3, 4))
print(xr)

# radom normal
xrn = torch.randn((3, 4))
print(xr)

# arange
ar = torch.arange(2, 20)  # torch.arange(20)
print(ar)

# linspace
lp = torch.linspace(
    0, 1, 5
)
print(lp)

# convert from numpy
a = np.array([1, 2, 3])
x = torch.from_numpy(a)
print(x)
# to numpy
a = x.numpy()
print(a)

# clone
x = torch.tensor([1, 2, 3])
y = x.clone()
print(y)

# reshape
x = torch.arange(12)
x = x.reshape(3, 4)
print(x)

# view
"""
In most cases, it is similar to `reshape`, but there is an important difference:

`view()` only works on tensors that are contiguous in memory.
`reshape()` creates a copy of the data if necessary.

In modern code, `reshape()` is usually the safer choice.
"""
x = x.view(3, 4)
print(x)

# unsqueeze , add dimension
x = torch.tensor([1, 2, 3])
print(x.shape)

x = x.unsqueeze(0)
print(x.shape)
# convert [1,2,3] -> [[1,2,3]]

# slice
print(x[2:6])

# nd
x = torch.arange(12).reshape(3,4)
print(x[1])
print(x[:,2])
print(x[0:2,1:3])