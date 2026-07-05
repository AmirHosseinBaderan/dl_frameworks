import tensorflow as tf
import numpy as np

w = tf.Variable([1, 2, 3])
print(w)

# assign
w.assign([4, 5, 6])
print(w)

# assign add
w.assign_add([7, 8, 9])
print(w)

# assign sub
w.assign_sub([1, 1, 1])
print(w)

# convert to numpy
print(w.numpy())

# from tensor to variable
t = tf.constant([1, 2, 3])
v = tf.Variable(t)
print(v)

# from numpy to variable
a = np.array([[1, 2, 3], [4, 5, 6]])
v = tf.Variable(a)
print(v)

# operations like constant
print(w * 2)
print(tf.reduce_sum(w))

# What elements in a neural network are variables?
# Almost all learnable parameters: Weights,Bias,Embedding,Convolution Kernel

# What element are not variables?
# Input,Labels,Dataset,Predictions

print(tf.rank(w).numpy())
print(tf.reduce_sum(w).numpy())