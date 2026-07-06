import tensorflow as tf

# how open tensorboard app
# tensorboard --logdir ./logs -> open localhost:6006

# how write manual logs
writer = tf.summary.create_file_writer('./logs/custom')

with writer.as_default():
    tf.summary.scaler(
        "loss",
        0.52,
        step=1,
    )

# manual histogram
"""
tf.summary.histogram(

    "weights",

    model.trainable_variables[0],

    step=epoch
)
"""

# image
# tf.summary.image(...)

