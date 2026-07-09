# FashionMNIST Classifier — PyTorch

## Project Description
A complete, end-to-end image-classification project built with **PyTorch**. It trains a
feed-forward neural network to classify the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
dataset (10 classes of clothing) into T-shirts, trousers, shoes, etc. The project is
split into small, single-responsibility modules: data loading, model definition,
training, evaluation, and prediction.

The same task is implemented in TensorFlow under [`../tensorflow`](../tensorflow/README.md)
so you can compare the two frameworks directly.

## How it works
1. **Data** (`dataset.py`): downloads FashionMNIST with `torchvision`, wraps train/test
   sets in `DataLoader`s (`BATCH_SIZE = 64`).
2. **Model** (`model.py`): `FashionClassifier(nn.Module)` — three `Linear` layers
   (784→128→64→10) with ReLU, flattening the `28×28` image first.
3. **Train** (`train.py`): a manual training loop — `zero_grad → forward → loss →
   backward → step` — for 10 epochs with `CrossEntropyLoss` + `Adam(lr=0.001)`. It also
   validates each epoch and saves `../models/fashion_classifier.pth`.
4. **Evaluate** (`evaluate.py`): loads the saved weights, computes test accuracy, and
   draws a confusion matrix with seaborn.
5. **Predict** (`predict.py`): loads the model, classifies a single test image, and
   displays it with the predicted vs actual label.
6. **Test** (`test_model.py`): a quick shape-check that the model runs on a random batch.

> Note: `utils.py` exists in this folder but is currently empty (reserved for helpers).

## Files
| File | Description |
|------|-------------|
| [`dataset.py`](dataset.py) | `load_dataset()` — downloads FashionMNIST and returns train/test `DataLoader`s. |
| [`model.py`](model.py) | `FashionClassifier` — the `nn.Module` with three linear layers. |
| [`train.py`](train.py) | `train_one_epoch`, `validate`, and `train` — the full training + validation loop; saves weights. |
| [`evaluate.py`](evaluate.py) | Loads weights, prints test accuracy, and plots a confusion matrix. |
| [`predict.py`](predict.py) | Loads weights and predicts a single image, showing prediction vs ground truth. |
| [`test_model.py`](test_model.py) | Sanity check: runs the model on a random `(64,1,28,28)` batch and prints output shape. |
| `utils.py` | (Empty) Reserved for shared helper functions. |

## Usage
```bash
cd project/pytorch
python train.py        # train and save ../models/fashion_classifier.pth
python evaluate.py     # accuracy + confusion matrix
python predict.py      # classify one sample image
python test_model.py   # shape sanity check
```
