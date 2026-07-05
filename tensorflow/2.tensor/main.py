import tensorflow as tf
import numpy as np

# scaler
scaler = tf.constant(42)

# vector
vector = tf.constant([1, 2, 3])

# matrix
matrix = tf.constant(
    [
        [1, 2],
        [3, 4]
    ]
)

# 3D tensor
tensor3D = tf.constant([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

# print info
for name, t in [
    ("scaler", scaler),
    ("vector", vector),
    ("matrix", matrix),
    ("tensor3D", tensor3D)
]:
    print("=" * 40)
    print(name)
    print(t)
    print("Shape : ", t.shape)
    print("Rank : ", tf.rank(t).numpy())
    print("Size : ", tf.size(t).numpy())
    print("Dtype : ", t.dtype)

# Numpy -> Tensor
arr = np.array([1, 2, 3])
tensor = tf.constant(arr)

print("\n Tensor from Numpy : ")
print(tensor)

# Tensor to Numpy
print("\n Back to numpy : ")
print(tensor.numpy())
