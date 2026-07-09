import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from dataset import load_dataset



def evaluate():
    _, test_ds = load_dataset()
    model = tf.keras.models.load_model(
        "../models/fashion_classifier.keras"
    )
    loss, accuracy = model.evaluate(
        test_ds
    )
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        logits = model(images)
        predictions = tf.argmax(
            logits,
            axis=1
        )
        y_true.extend(
            labels.numpy()
        )
        y_pred.extend(
            predictions.numpy()
        )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(10,8)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.show()

if __name__ == "__main__":
    evaluate()