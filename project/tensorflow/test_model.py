from model import build_model
import tensorflow as tf

model = build_model()

model.summary()

x = tf.random.normal((1,28,28))

output = model(x)

print(output.shape)