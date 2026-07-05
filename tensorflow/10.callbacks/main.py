import tensorflow as tf
from custom import MyCallback

# callback -> Execute a piece of code at specific times during training.

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='./best.keras',
    save_best_only=True,  # if this be false , call back save all epochs, if true then save only best epoch
    monitor='val_loss',  # Which criterion should be examined? loss,accuracy,va_accuracy,val_loss
)

## model.fit(....,callbacks=[checkpoint])

early = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # If 5 epochs pass without any progress, stop training.
    restore_best_weights=True
    # Suppose the best model was at Epoch 20, but early stopping occurred at Epoch 30. If this option is enabled, the weights from Epoch 20 are restored.
)

# In other words, if there is no improvement after 3 epochs, reduce the learning rate by a factor of 10.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
)

# save all losses in csv
csv_logger = tf.keras.callbacks.CSVLogger(
    "./training.csv"
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir="./logs",
)

# model.fit(
#
#     train_dataset,
#
#     validation_data=valid_dataset,
#
#     epochs=100,
#
#     callbacks=[
#
#         checkpoint,
#
#         early,
#
#         reduce_lr,
#
#         tensorboard,
#         MyCallback()
#     ]
# )

# The best combination
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