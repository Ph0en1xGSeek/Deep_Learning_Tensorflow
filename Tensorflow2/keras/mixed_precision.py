import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time


policy = mixed_precision.Policy('mixed_float16') # mixed precision
mixed_precision.set_policy(policy)
loss_scale = policy.loss_scale
print("Loss scale: %s" % loss_scale)

print("Compute dtype: %s" % policy.compute_dtype) # activation precision
print("Variable dtype: %s" % policy.variable_dtype) # weights precision

inputs = keras.Input(shape=(784, ), name='digits')
if tf.config.list_physical_devices('GPU'):
    print("The model will run with 4096 units on a GPU")
    num_units = 4096
else:
    print("The model will run with 64 units on a CPU")
    num_units = 64

dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='desne_2')
x = dense2(x)

print('x.dtype: %s' % x.dtype.name)
print('dense.kernel.dtype: %s' % dense1.kernel.dtype.name)

x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x) # use float32 in last layer
print('Outputs dtype: %s' % outputs.dtype.name)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
                optimizer=keras.optimizers.RMSprop(),
                metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

initial_wieghts = model.get_weights()

history = model.fit(x_train, y_train, batch_size=8192, epochs=5, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(8192))
test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(8192))

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
        scaled_loss = optimizer.get_scaled_loss(loss) # multiplies the loss by loss scale
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients) # divide the gradients by loss scale
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(x):
    return model(x, training=False)

model.set_weights(initial_wieghts)

for epoch in range(5):
    start = time.time()
    epoch_loss_avg = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy'
    )
    for x, y in train_dataset:
        loss = train_step(x, y)
        epoch_loss_avg(loss)
    for x, y in test_dataset:
        predictions = test_step(x)
        test_accuracy.update_state(y, predictions)
    print('Epoch {}: loss={}, test accuracy={}, time {}'.format(epoch, epoch_loss_avg.result(), test_accuracy.result(), time.time() - start))
