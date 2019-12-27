from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train, x_test, y_test = x_train[:1000], y_train[:1000], x_test[:1000], y_test[:1000]
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1,
                                                period=5)
model = create_model()

model.save_weights(checkpoint_path.format(epoch=66666)) # saving without callback

model.fit(x_train, y_train, epochs=15,
            validation_data=(x_test, y_test),
            callbacks=[cp_callback])
model.evaluate(x_test, y_test, verbose=2)



# Load

model = create_model()

loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

latest = tf.train.latest_checkpoint(checkpoint_dir)
print("latest", latest)
model.load_weights(checkpoint_path.format(epoch=15))

loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

model.save('my_model.h5')
new_model = tf.keras.models.load_model('my_model.h5')
loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
print("Restored new model, accuracy: {:5.2f}%".format(100 * acc))

