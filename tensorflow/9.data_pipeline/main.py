import tensorflow as tf
from numpy import random

x = [1, 2, 3, 4]

dataset = tf.data.Dataset.from_tensor_slices(x)

for i in dataset:
    print(i.numpy())

x = random.randint(100, size=(3, 5))
dataset = tf.data.Dataset.from_tensor_slices(x)
for i in dataset:
    print(i.numpy())

dataset = dataset.shuffle(1000)
for i in dataset:
    print(i.numpy())

dataset = (
    dataset
    .shuffle(1000)
    .batch(10)
)
for i in dataset:
    print(i.numpy())

# dont finish dataset
# dataset = dataset.repeat()
for i in dataset:
    print(i.numpy())


def normalize(x):
    x = x / 255

    return x


# enable function on data set
dataset = dataset.map(normalize)


def square(x):
    return x * x


dataset = dataset.map(square)

# prefetch
# cpu and gpu work togather (cpu make next batch , gpu train last batch)
dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)
for i in dataset:
    print(i.numpy())

# cache, if dataset is static , for example MNIST, do not read the file again , during each epoch

# pipeline
dataset = (
    tf.data.Dataset
    .from_tensor_slices(x)
    .shuffle(1000)
    .map(normalize)
    .batch(10)
    .prefetch(tf.data.AUTOTUNE)
)

# using data pipeline in side training loop
# instead model(x)
"""
for batch_x, batch_y in dataset:

    with tf.GradientTape() as tape:

        prediction = model(batch_x)

        loss = loss_fn(
            batch_y,
            prediction
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )
"""

# Standard pipeline sequence
dataset = (
    tf.data.Dataset
    .from_tensor_slices(x)
    .cache()
    .shuffle(1000)
    .map(normalize)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

x = tf.range(20)

dataset = (
    tf.data.Dataset
    .from_tensor_slices(x)
    .shuffle(20)
    .map(square)
    .batch(4)
)

for batch in dataset:
    print(batch.numpy())