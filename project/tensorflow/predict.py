import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def predict():
    model = tf.keras.models.load_model(
        "../models/fashion_classifier.keras"
    )

    (_, _), (test_images, test_labels) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    image = test_images[0]

    processed_image = (
        image
        .astype("float32")
        /
        255.0
    )

    processed_image = np.expand_dims(
        processed_image,
        axis=0
    )

    logits = model.predict(
        processed_image
    )

    predicted_class = np.argmax(
        logits
    )

    print(
        "Prediction:",
        class_names[predicted_class]
    )
    print(
        "Actual:",
        class_names[test_labels[0]]
    )

    plt.imshow(
        image,
        cmap="gray"
    )
    plt.title(
        class_names[predicted_class]
    )
    plt.show()

if __name__ == "__main__":
    predict()