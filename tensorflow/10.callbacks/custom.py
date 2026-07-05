import tensorflow as tf


class MyCallback(
    tf.keras.callbacks.Callback
):
    def on_epoch_end(self, epoch, logs=None):
        print(
            epoch,
            logs["loss"],
        )

        print(
            f"loss : {logs['loss']} / accuracy : {logs['accuracy']} / val_loss : {logs['val_loss']} / val_accuracy : {logs['val_accuracy']}")

        if logs["accuracy"] > 0.99:
            self.model.stop_training = True
