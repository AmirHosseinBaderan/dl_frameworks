import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b
print(c)

c = torch.add(a, b)
print(c)

# minus
c = a - b
print(c)
c = torch.sub(a, b)
print(c)

# Multiplication
c = a * b
print(c)
c = torch.mul(a, b)
print(c)

# Division
c = a / b
print(c)
c = torch.div(a, b)
print(c)

# power
x = torch.tensor([2, 3, 4])
print(x ** 2)
p = torch.pow(x, 2)
print(p)

# Square root
s = torch.sqrt(x)
print(s)

# log
l = torch.log(x)
print(l)

# exponential
e = torch.exp(x)
print(e)

# abs,Absolute value
a = torch.abs(x)
print(a)

# sin
s = torch.sin(x)
print(s)

# Cosine
c = torch.cos(x)
print(c)

# Tangent
t = torch.tan(x)
print(t)

# mean
m = x.mean(dtype=torch.float32)
print(m)

# sum
s = x.sum()
print(s)

# max
m = x.max()
print(m)

# min
m = x.min()
print(m)

# max index
ma = x.argmax()
print(ma)

# min index
ma = x.argmin()
print(ma)

# Matrix Multiplication
A = torch.tensor([
    [1, 2],
    [3, 4]
], dtype=torch.float32)

B = torch.tensor([
    [5, 6],
    [7, 8]
], dtype=torch.float32)

C = A @ B
print(C)

print(C)
# or
c = torch.matmul(A, B)
print(c)

# Transpose
print(A.T)
print(torch.transpose(A, 0, 1))

# re shape
r = x.reshape((1, 3))
print(r)

# flatten
x = torch.tensor([
    [1, 2],
    [3, 4]
], dtype=torch.float32)
f = x.flatten()
print(f)

# concatenate
a = torch.tensor([
    [1, 2],
], dtype=torch.float32)
b = torch.tensor([[3, 4]], dtype=torch.float32)

c = torch.cat([a, b])

# stack , It adds a new dimension.
s = torch.stack([a, b])
print(s)

# broadcasting
# if we have [1,2,3]
# x + 1 -> [2,3,4]
# or [[1,2],[3,4]]
# x + 1 -> [[2,3],[4,5]]
A = torch.ones((1, 2))
b = torch.tensor([3, 4])
print(A + b)

# Comparison , out put is tensor of type Boolean
print(a > b)
print(a < b)
print(a == b)

# Masking
x = torch.arange(10)
mask = x > 5
print(mask)  # -> False False False False False False True True True True
print(x[mask])  # -> 6 7 8 9

# in place operations
x.add_(1)
# its like x += 1

# device
# x = x.to("cuda") -> change operation to GPU
# x = x.to("cpu") -> Change to CPU

ts = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
).float()

print("ts sum : ", ts.sum())
print("ts mean : ", ts.mean(dtype=torch.float32))
print("ts mean : ", ts.float().mean())
print("ts sum of dim 0 (cols)", ts.sum(dim=0))
print("ts sum of dim 1 (rows)", ts.sum(dim=1))
print("ts Transpose : ", ts.T)
print("ts flatten : ", ts.flatten())

print("larger than 5 with maskin")
mask = ts > 5
larger = ts[mask]
print("larger : ", larger)

print("ts mul : ", ts @ ts.T)
