import tensorflow as tf
from custom_layers import DoubleLayer, MyLayer, MyDense, MyDenseWithActivation, MyLinear
import numpy as np

layer = DoubleLayer()
x = tf.constant([1, 2, 3])

my_layer = MyLayer()

print(layer(x))
print(my_layer(x))

# use custom layers in model
model = tf.keras.Sequential([
    MyDense(),
    MyDenseWithActivation(),
    tf.keras.layers.Dense(10),
])

y = model(np.random.normal(0, 1, (10, 10)))

print(model.trainable_variables)
print(model.summary())

model2 = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    MyLinear(128),
    tf.keras.layers.ReLU(),
    MyLinear(10),
    tf.keras.layers.Softmax()
])

model.summary()
