import tensorflow as tf

x = tf.ones((2, 2))

with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# when persistent is set to False
# gradient can only be called once, after that, resource of GradientTape will be released
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0

dz_dy = t.gradient(z, y)
print(dz_dy.numpy())
assert dz_dy.numpy() == 8.0

def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)


if __name__ == "__main__" :
    x = tf.convert_to_tensor(2.0)

    assert grad(x, 6).numpy() == 12.0
    assert grad(x, 5).numpy() == 12.0
    assert grad(x, 4).numpy() == 4.0

    x = tf.Variable(1.0)

    # # Calling GradientTape.gradient on a persistent tape inside its context is significantly less efficient than calling it outside the context 
    # # (it causes the gradient ops to be recorded on the tape, leading to increased CPU and memory usage)
    # with tf.GradientTape(persistent=True) as t:
    #     y = x * x * x
    #     dy_dx = t.gradient(y, x)
    # d2y_dx = t.gradient(dy_dx, x)

    with tf.GradientTape() as t:
        with tf.GradientTape() as t2:
            y = x * x * x
        dy_dx = t2.gradient(y, x)
    d2y_dx = t.gradient(dy_dx, x)

    print(dy_dx.numpy(), d2y_dx.numpy())
    assert dy_dx.numpy() == 3.0
    assert d2y_dx.numpy() == 6.0




