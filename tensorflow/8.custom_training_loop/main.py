# when we write model.fit(x_train,y_trian)
# TensorFlow performs these tasks, more or less:
# for epoch:
#
#     for batch:
#
#         Forward
#
#         Loss
#
#         Backward
#
#         Update Weights

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])
# loss
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
)

x = tf.constant([
    [1., 2.],
    [2., 3.],
    [3., 4.],
    [4., 5.]
])

y = tf.constant([
    [3.],
    [5.],
    [7.],
    [9.]
])
# training step
epochs = 100

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        prediction = model(x)

        loss = loss_fn(
            y,
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

    print(
        epoch,
        loss.numpy()
    )
