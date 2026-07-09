# dl_frameworks

A hands-on learning repository that teaches **PyTorch** and **TensorFlow** side by side.
Each framework has a step-by-step tutorial track (folders `1.*` → `13.*`) that builds
concepts from the most basic tensor up to saving/loading models, plus a complete
end-to-end **FashionMNIST classification project** implemented in both frameworks so you
can compare the same task written two different ways.

## Repository Structure

```
dl_frameworks/
├── README.md                 # This file (overview + TF vs PyTorch)
├── requirements.txt          # Pinned dependencies (torch, tensorflow, keras, ...)
├── pytorch/                  # PyTorch tutorial track (13 steps)
│   ├── overview.txt          # Quick concept-mapping cheat sheet
│   ├── 1.step1/  ...  13.saving_loading_model/
├── tensorflow/               # TensorFlow tutorial track (12 steps)
│   ├── 1.step1/  ...  12.save_load_model/
└── project/                  # Full FashionMNIST project in both frameworks
    ├── data/                 # Downloaded FashionMNIST dataset
    ├── models/               # Trained model artifacts (.pth / .keras)
    ├── pytorch/              # PyTorch implementation
    └── tensorflow/           # TensorFlow implementation
```

Every tutorial folder contains its own `README.md` that explains the goal of the step,
how the code works, and a description of each file inside it.

### PyTorch tutorial track

| Step | Topic | Link |
|------|-------|------|
| 1  | Setup / verify install | [`pytorch/1.step1`](pytorch/1.step1/README.md) |
| 2  | Tensors & operations | [`pytorch/2.tensor`](pytorch/2.tensor/README.md) |
| 3  | Autograd (automatic differentiation) | [`pytorch/3.autograd`](pytorch/3.autograd/README.md) |
| 4  | `nn.Module` (build a model) | [`pytorch/4.module`](pytorch/4.module/README.md) |
| 5  | Forward pass | [`pytorch/5.forward_pass`](pytorch/5.forward_pass/README.md) |
| 6  | `Dataset` | [`pytorch/6.dataset`](pytorch/6.dataset/README.md) |
| 7  | `DataLoader` | [`pytorch/7.data_loader`](pytorch/7.data_loader/README.md) |
| 8  | Loss functions | [`pytorch/8.loss_functions`](pytorch/8.loss_functions/README.md) |
| 9  | Optimizers | [`pytorch/9.optimizer`](pytorch/9.optimizer/README.md) |
| 10 | Custom training loop | [`pytorch/10.custom_training_loop`](pytorch/10.custom_training_loop/README.md) |
| 11 | GPU training | [`pytorch/11.gpu_training`](pytorch/11.gpu_training/README.md) |
| 12 | TorchVision & datasets | [`pytorch/12.torch_vision`](pytorch/12.torch_vision/README.md) |
| 13 | Saving / loading models | [`pytorch/13.saving_loading_model`](pytorch/13.saving_loading_model/README.md) |

### TensorFlow tutorial track

| Step | Topic | Link |
|------|-------|------|
| 1  | Setup / verify install | [`tensorflow/1.step1`](tensorflow/1.step1/README.md) |
| 2  | Tensors & operations | [`tensorflow/2.tensor`](tensorflow/2.tensor/README.md) |
| 3  | `tf.Variable` | [`tensorflow/3.variable`](tensorflow/3.variable/README.md) |
| 4  | `GradientTape` (automatic differentiation) | [`tensorflow/4.gradient_tape`](tensorflow/4.gradient_tape/README.md) |
| 5  | Keras Sequential API | [`tensorflow/5.keras_sequential_api`](tensorflow/5.keras_sequential_api/README.md) |
| 6  | Functional API | [`tensorflow/6.functional_api`](tensorflow/6.functional_api/README.md) |
| 7  | Custom layers | [`tensorflow/7.custom_layer`](tensorflow/7.custom_layer/README.md) |
| 8  | Custom training loop | [`tensorflow/8.custom_training_loop`](tensorflow/8.custom_training_loop/README.md) |
| 9  | `tf.data` pipeline | [`tensorflow/9.data_pipeline`](tensorflow/9.data_pipeline/README.md) |
| 10 | Callbacks | [`tensorflow/10.callbacks`](tensorflow/10.callbacks/README.md) |
| 11 | TensorBoard | [`tensorflow/11.tensor_board`](tensorflow/11.tensor_board/README.md) |
| 12 | Saving / loading models | [`tensorflow/12.save_load_model`](tensorflow/12.save_load_model/README.md) |

### End-to-end project

| Framework | Link |
|-----------|------|
| PyTorch FashionMNIST project | [`project/pytorch`](project/pytorch/README.md) |
| TensorFlow FashionMNIST project | [`project/tensorflow`](project/tensorflow/README.md) |

## Installation

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins `torch`, `torchvision`, `tensorflow`, `keras`, `numpy`,
`matplotlib`, `seaborn`, `scikit-learn`, and the CUDA toolchain so the environment is
reproducible.

---

# TensorFlow vs PyTorch

Both frameworks can solve the same problems, but they take very different philosophical
approaches. The table below maps the core concepts 1-to-1.

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

## Creating a Model

```python
# tensorflow
model = tf.keras.Sequential([
    Dense(64),
    Dense(10)
])
```

```python
# pytorch
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

## Model Training

Here we see the first major difference.

```python
# tensorflow
model.fit(
    train_dataset,
    epochs=10
)
```

```python
# pytorch
for epoch in range(epochs):
    for x, y in loader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
```

### TensorFlow

```
Pros:
✅ Less code
✅ Rapid development
✅ Excellent deployment
✅ Keras
✅ Production-ready

Disadvantages:
❌ Less control
❌ Harder to customize
```

### PyTorch

```
Pros:
✅ Full control
✅ Easy debugging
✅ Suitable for research
✅ Pythonic coding

Disadvantages:
❌ More code
❌ You have to write a Training Loop
```

```
If you want to create a new research algorithm—
for instance, you've read a paper that says:

"Modify the loss function like this."
In PyTorch:
It's very easy.
In TensorFlow:
It's a bit harder.
```

## Quick concept cheat sheet

| TensorFlow                 | PyTorch                    |
| -------------------------- | -------------------------- |
| `tf.Tensor`                | `torch.Tensor`             |
| `tf.Variable`              | `nn.Parameter`             |
| `Dense`                    | `Linear`                   |
| `Sequential`               | `nn.Sequential`            |
| `Model`                    | `nn.Module`                |
| `GradientTape`             | `loss.backward()`          |
| `model.fit()`              | manual training loop       |
| `tf.data.Dataset`          | `torch.utils.data.Dataset` |
| `tf.keras.optimizers.Adam` | `torch.optim.Adam`         |

## When to choose which?

- **Choose TensorFlow / Keras** when you want to ship a model to production quickly,
  leverage high-level `model.fit()` training, built-in callbacks, TensorBoard, and
  robust deployment tooling (TF Serving, TFLite, TF.js).
- **Choose PyTorch** when you are doing research, need fine-grained control over the
  training loop, custom loss functions, or want a more "Pythonic" and debuggable
  experience.
