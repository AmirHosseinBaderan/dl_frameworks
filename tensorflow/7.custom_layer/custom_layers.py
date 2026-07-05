import tensorflow as tf


class DoubleLayer(tf.keras.Layer):

    def call(self, inputs):
        return inputs * 2


class MyLayer(tf.keras.Layer):
    def __init__(self):
        super().__init__()

        self.w = self.add_weight(
            shape=(),
            initializer='ones',
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.w


# sample , when we dont know input size
class MyDense(tf.keras.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 64),
            initializer='random_normal',
        )

        self.b = self.add_weight(
            shape=(64,),
            initializer='zeros',
        )

    def call(self, inputs):
        return tf.matmul(
            inputs,
            self.w,
        ) + self.b


class MyDenseWithActivation(tf.keras.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 64),
            initializer='random_normal',
        )

        self.b = self.add_weight(
            shape=(64,),
            initializer='zeros',
        )

    def call(self, inputs):
        inputs = tf.matmul(
            inputs,
            self.w,
        ) + self.b

        return tf.nn.relu(inputs)


class AddTen(tf.keras.Layer):
    def call(self, inputs):
        return inputs + 10


class Square(tf.keras.Layer):
    def call(self, inputs):
        return inputs ** 2


class MyLinear(tf.keras.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name="kernel"
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
