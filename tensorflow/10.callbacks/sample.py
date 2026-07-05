import tensorflow as tf
import numpy as np

x_train = np.random.random((1000, 128)).astype(np.float32)

y_train = tf.keras.utils.to_categorical(
    np.random.randint(0, 10, size=(1000,)),
    num_classes=10
)

print(x_train)
print(x_train.shape)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        "best.keras",
        save_best_only=True,
        monitor="val_loss"
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5
    ),

    tf.keras.callbacks.TensorBoard(
        log_dir="logs"
    )
]

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    callbacks=callbacks
)

predict = model.predict(np.random.random((1,128)).astype(np.float32))
print(predict)