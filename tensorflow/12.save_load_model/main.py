# easy way
# save model
# model.save("./model/my_model.keras")

# load saved model
# loaded_model = tf.keras.models.load_model("./model/my_model.keras")

# save and load only weights
# model.save_weights("./weights.weights.h5")
# model.load_weights("./weights.weights.h5")

"""
When should you use `save_weights`?
For instance,
when the architecture
is already in Git.
Only the weights
need to be downloaded.

When should you use `model.save`?
Almost always.
Because everything gets saved.
"""

# model checkpoint do this (model.save)

import tensorflow as tf
import numpy as np

x = np.random.random((1000, 128))
y = tf.keras.utils.to_categorical(np.random.randint(0, 10, 1000))

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x,
    y,
    epochs=5
)

model.save("./demo.keras")

loaded = tf.keras.models.load_model(
    "demo.keras"
)

print(
    loaded.predict(x[:3])
)

# if we have custom layer then in load most say to keras whats that
"""
loaded = tf.keras.models.load_model(
    "demo.keras",
    custom_objects={
        "MyDense": MyDense
    }
)
"""

# if we have custom loss then
"""
custom_objects={
    "MyLoss": MyLoss
}
"""