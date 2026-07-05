import tensorflow as tf

# gradient tape : Write down every action you perform so that I can calculate its derivative later.

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

gradient = tape.gradient(y, x)
print(gradient)

x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = 5 * x + 3

gradient = tape.gradient(y, x)
print(gradient)

x = tf.Variable(4.0)
with tf.GradientTape() as tape:
    y = x ** 3

gradient = tape.gradient(y, x)
print(gradient)

# Multi variables
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = x * y

gradient = tape.gradient(z, [x, y])
print("Gradient z on [x,y]")
for g in gradient:
    print(g.numpy())

# gradient tape only make gradient on Element that are learnable -> tf.Variable
# if we wanna make gradient on constant we most call watch

x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2

gradient = tape.gradient(y, x)
print(gradient)

# So, by default, GradientTape only tracks tf.Variables.

w = tf.Variable(2.0)

with tf.GradientTape() as tape:
    loss = (w - 5) ** 2

grad = tape.gradient(loss, w)

print("Weight :", w.numpy())
print("Loss   :", loss.numpy())
print("Grad   :", grad.numpy())

learning_rate = 0.1
w.assign_sub(learning_rate * grad)
print("Updated Weight : ", w.numpy())
