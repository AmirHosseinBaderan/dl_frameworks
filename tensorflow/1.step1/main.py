import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
print(tf.config.list_physical_devices())