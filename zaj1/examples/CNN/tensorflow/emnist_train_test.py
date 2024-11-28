import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

print(tf.__version__)
model = tf.keras.models.Sequential(
    [
        # LENET5 https://engmrk.com/lenet-5-a-classic-cnn-architecture/
        # tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Conv2D(10, kernel_size=(5, 5), strides=(1, 1),
                               input_shape=(28, 28,1), activation=tf.nn.tanh),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(60, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.tanh),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.tanh),  # tf.nn.relu
        tf.keras.layers.Dense(47, activation=tf.nn.softmax)
    ]
)

model.compile(optimizer=tf.keras.optimizers.SGD(0.01),#GradientDescentOptimizer
              loss = tf.keras.losses.sparse_categorical_crossentropy, #tf.kearas.losses.mean_absolute_error
              metrics=['accuracy']
              )

model.summary()


mnist =tf.keras.datasets.mnist

(ds_train, ds_test), ds_info = tfds.load('emnist/balanced', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)
ds_train.batch(64)
ds_test.batch(64)
history = model.fit(ds_train,
                    epochs=10, batch_size=64, verbose=1
                   )
test_loss, test_acc = model.evaluate(ds_test)
print('Test acc:',test_acc)
print('Test loss:', test_loss)
fig = plt.figure()
plt.plot([*range(len(history.history['loss']))], history.history['loss'], color='blue')
plt.plot([*range(len(history.history['val_loss']))], history.history['val_loss'], color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('sparse categorical crossentropy loss')
fig.show()