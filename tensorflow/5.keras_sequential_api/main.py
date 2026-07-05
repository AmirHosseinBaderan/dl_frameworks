import tensorflow as tf

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10),
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        128,
        activation="relu",
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu",
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax",
    )
])

# build model
# 1
model.build((None, 784))

# 2
# model(tf.zeros(1, 784))

print(model.summary())

# forward
x = tf.random.normal((5, 784))
y = model(x)
print(y.shape)

# input layer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),

    tf.keras.layers.Dense(
        128,
        activation="relu",
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu",
    )
])
print("model before add layer")
print(model.summary())

# This allows the model to know the input shape right from the start, enabling `summary()` to work without requiring an additional build step.

model.add(
    tf.keras.layers.Dense(
        10,
        activation="softmax",
    )
)
print("Add layer")
model.summary()
# check layers and models

model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

x = tf.random.normal((8, 784))

y = model(x)

print("\nOutput Shape:", y.shape)
print("\nFirst Prediction:")
print(y[0].numpy())
print("\nSum of probabilities:", tf.reduce_sum(y[0]).numpy())