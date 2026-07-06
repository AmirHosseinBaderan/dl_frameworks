import torch

x = torch.tensor(
    2.0,
    requires_grad=True
)

y = 3 * x + 5

y.backward()

print(x.grad)

# chain rule
a = x + 1
b = a ** 2
c = b * 3
print(c)
c.backward()
print(x.grad)

# this work only on scaler
# for vector most use torch funcs

x2 = torch.tensor(
    [1., 2., 3.],
    requires_grad=True
)

y2 = x2 * 2

loss = y2.mean()
loss.backward()

print(x2.grad)

w = torch.tensor(
    2.0,
    requires_grad=True
)

x = torch.tensor(3.0)

y_true = torch.tensor(10.0)

prediction = w * x

loss = (prediction - y_true) ** 2

loss.backward()

print(w.grad)