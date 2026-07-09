import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    return model
