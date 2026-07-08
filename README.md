# dl_frameworks

| Concept     | TensorFlow            | PyTorch        |
|-------------|-----------------------|----------------|
| Tensor      | `tf.Tensor`           | `torch.Tensor` |
| Variable    | `tf.Variable`         | `nn.Parameter` |
| Model       | `keras.Model`         | `nn.Module`    |
| Dataset     | `tf.data.Dataset`     | `Dataset`      |
| Data Loader | `tf.data`             | `DataLoader`   |
| Loss        | `tf.keras.losses`     | `nn.*Loss`     |
| Optimizer   | `tf.keras.optimizers` | `torch.optim`  |
| Save        | `model.save()`        | `torch.save()` |
| Predict     | `model.predict()`     | `model(x)`     |

# Create Model

``` Python
# tensorflow
model = tf.keras.Sequential([
    Dense(64),
    Dense(10)
])
```

``` Python
# pytorch
model = nn.Sequential(
    nn.Linear(784,64),
    nn.ReLU(),
    nn.Linear(64,10)
)
```

# Model Training

#### Here, we see the first major difference.

``` Python
# tensorflow
model.fit(
    train_dataset,
    epochs=10
)
```

``` Python
# pytorch
for epoch in range(epochs):

    for x,y in loader:

        optimizer.zero_grad()

        prediction = model(x)

        loss = criterion(prediction,y)

        loss.backward()

        optimizer.step()
```

### TensorFlow

```
Pros :
✅ Less code
✅ Rapid development
✅ Excellent deployment
✅ Keras
✅ Production-ready

Disadvantages :
❌ Less control
❌ Harder to customize
```

### Pytorch

```
Pros:
✅ Full control
✅ Easy debugging
✅ Suitable for research
✅ Pythonic coding

Disadvantages :
❌ More code
❌ You have to write a Training Loop
```

```
If you want to create a new research algorithm—
for instance, you’ve read a paper that says:

"Modify the loss function like this."
In PyTorch:
It’s very easy.
In TensorFlow:
It’s a bit harder.
```