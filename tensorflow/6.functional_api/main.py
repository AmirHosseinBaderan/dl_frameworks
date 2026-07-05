import tensorflow as tf

"""
What is the Functional API?

In the Functional API,

instead of listing the layers,

we specify

where the output of each layer connects.

It is exactly like drawing a graph.
"""

inputs = tf.keras.Input(shape=(784,))

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(inputs)

outputs = tf.keras.layers.Dense(
    10,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)

# make branches
"""
           Input
          /     \\
     Dense64   Dense64
          \     /
      Concatenate
             │
         Dense10
"""
inputs = tf.keras.Input(shape=(100,))

branch1 = tf.keras.layers.Dense(64)(inputs)
branch2 = tf.keras.layers.Dense(64)(inputs)

merged = tf.keras.layers.Concatenate()([branch1, branch2])

outputs = tf.keras.layers.Dense(10)(merged)

# multi inputs
image = tf.keras.layers.Input(shape=(28, 28, 1))
age = tf.keras.layers.Input(shape=(1,))

# Sequential vs Functional
"""
    Sequential
    Input

↓

↓

↓

↓

Output
"""

"""
Functional
           Input

        /     |     \\

     CNN   Dense   LSTM

        \     |     /

      Concatenate

            |

         Output
"""

inputs = tf.keras.Input(shape=(32,))

left = tf.keras.layers.Dense(64, activation="relu")(inputs)
right = tf.keras.layers.Dense(64, activation="relu")(inputs)

merged = tf.keras.layers.Concatenate()([left, right])

outputs = tf.keras.layers.Dense(10, activation="softmax")(merged)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()