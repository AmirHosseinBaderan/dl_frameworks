import tensorflow as tf

# element-wise operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

c = a + b
print(c)

c = a * b
print(c)

c = a - b
print(c)

c = a / b
print(c)

# none linear functions
x = tf.constant([-1.0, 0.0, 1.0])
print(tf.nn.relu(x))

# Reduction Operations
# convert multi values to one values
x = tf.constant([
    [1, 2],
    [3, 4]
])
print(tf.reduce_sum(x))
print(tf.reduce_sum(x, axis=0))  # columns
print(tf.reduce_sum(x, axis=1))  # rows

# mean
print(tf.reduce_mean(x))

# min/max
print(tf.reduce_min(x))
print(tf.reduce_max(x))

# broadcasting
# The Golden Rule of Broadcasting
# If shapes are compatible for alignment → the operation proceeds.
a = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
])
b = tf.constant([1, 2, 3])
print(a + b)

# matrix operations
#matmul → matrix multiplication
a = tf.constant([
    [1, 2],
    [3, 4]
])
b = tf.constant([
    [5, 6],
    [7, 8]
])

print(tf.matmul(a,b))

# Slicing Indexing
x = tf.constant([10, 20, 30])

print(x[0])
print(x[1:3])

m = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
])

print(m[0, 1])
print(m[:, 1])

# type casting
x = tf.constant([1, 2, 3], dtype=tf.int32)
y = tf.cast(x,tf.float32)
print(y)

# shape manipulation
x = tf.constant([1, 2, 3, 4, 5, 6])