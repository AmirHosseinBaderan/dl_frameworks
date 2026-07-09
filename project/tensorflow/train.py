import tensorflow as tf

from dataset import load_dataset
from model import build_model


def train():
    train_ds, test_ds = load_dataset()
    model = build_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=[
            "accuracy"
        ]

    )
    model.summary()
    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=test_ds
    )
    model.save(
        "../models/fashion_classifier.keras"
    )

    return history



if __name__ == "__main__":
    train()