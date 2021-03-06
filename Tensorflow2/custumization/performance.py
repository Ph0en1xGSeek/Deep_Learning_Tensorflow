import tensorflow as tf
import traceback
import contextlib

@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n {}:'.format(error_class))
        traceback.print_exc(limit=2)
    except Exception as e:
        raise e
    else:
        raise Exception('Excepted {} to be raised but no error was raised!'.format(error_class))

@tf.function
def add(a, b):
    return a + b

# print(add(tf.ones([2, 2]), tf.ones([2, 2])))

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
# print(tape.gradient(result, v))

@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)

# print(dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2])))

@tf.function
def double(a):
    print("Tracing with", a) # print only on the first time (building trace)
    return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()
print(double(tf.constant("a")))
print()

print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(tf.constant("b")))
print("Using a concrete trace with incompatible types will throw an error")
with assert_raises(tf.errors.InvalidArgumentError):
    double_strings(tf.constant(1))

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32), ))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(x % 2 == 0, x // 2, 3 * x + 1)

print(next_collatz(tf.constant([1, 2])))

with assert_raises(ValueError):
    next_collatz(tf.constant([[1, 2], [3, 4]]))

def train_one_step():
    pass

@tf.function
def train(num_steps):
    print("Tracing with num_step = {}".format(num_steps))
    for _ in tf.range(num_steps):
        train_one_step()

# train(num_steps=10)
# train(num_steps=20)

train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))