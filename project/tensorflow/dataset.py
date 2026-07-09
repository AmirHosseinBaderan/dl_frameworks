import tensorflow as tf
import matplotlib.pyplot as plt

BUFFER_SIZE = 60000
BATCH_SIZE = 64


def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)
    )

    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)
    )

    train_ds = (
        train_ds
        .shuffle(60000)
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds


if __name__ == "__main__":
    train_ds, test_ds = load_dataset()

    for image, label in train_ds.take(1):
        print(image.shape)
        print(label)
