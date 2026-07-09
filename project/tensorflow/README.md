# FashionMNIST Classifier — TensorFlow

## Project Description
A complete, end-to-end image-classification project built with **TensorFlow / Keras**.
It trains a feed-forward neural network to classify the
[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset (10 classes of
clothing). The project is organized into small modules: data loading, model building,
training, evaluation, and prediction, mirroring the PyTorch version under
[`../pytorch`](../pytorch/README.md) for a side-by-side comparison.

## How it works
1. **Data** (`dataset.py`): loads FashionMNIST via `tf.keras.datasets`, normalizes pixel
   values to `[0,1]`, and builds `tf.data` pipelines (`BUFFER_SIZE=60000`,
   `BATCH_SIZE=64`) with `shuffle` + `prefetch`.
2. **Model** (`model.py`): `build_model()` returns a `keras.Sequential` — `Flatten(28,28)`
   → `Dense(128, relu)` → `Dense(64, relu)` → `Dense(10)` (logits, no softmax).
3. **Train** (`train.py`): `model.compile(optimizer="adam",
   loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])` then
   `model.fit(...)` for 10 epochs with validation; saves `../models/fashion_classifier.keras`.
4. **Evaluate** (`evaluate.py`): reloads the saved model, prints test accuracy, and plots
   a confusion matrix with seaborn.
5. **Predict** (`predict.py`): reloads the model, classifies a single test image, and
   displays it with the predicted vs actual label.
6. **Test** (`test_model.py`): prints the model summary and checks the output shape on a
   random `(1,28,28)` input.

> Note: `utils.py` exists in this folder but is currently empty (reserved for helpers).
> There is no `main.py`; the entry points are the individual `train.py`, `evaluate.py`,
> and `predict.py` scripts.

## Files
| File | Description |
|------|-------------|
| [`dataset.py`](dataset.py) | `load_dataset()` — loads and normalizes FashionMNIST, returns shuffled/prefetched `tf.data` pipelines. |
| [`model.py`](model.py) | `build_model()` — builds the `keras.Sequential` classifier. |
| [`train.py`](train.py) | Compiles and trains the model with `model.fit`, then saves it. |
| [`evaluate.py`](evaluate.py) | Loads the saved model, prints accuracy, and plots a confusion matrix. |
| [`predict.py`](predict.py) | Loads the saved model and predicts a single image, showing prediction vs ground truth. |
| [`test_model.py`](test_model.py) | Prints the model summary and verifies output shape on a random input. |
| `utils.py` | (Empty) Reserved for shared helper functions. |

## Usage
```bash
cd project/tensorflow
python train.py        # train and save ../models/fashion_classifier.keras
python evaluate.py     # accuracy + confusion matrix
python predict.py      # classify one sample image
python test_model.py   # model summary + shape check
```
