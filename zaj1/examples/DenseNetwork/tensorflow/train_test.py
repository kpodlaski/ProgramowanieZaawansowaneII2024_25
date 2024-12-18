import tensorflow as tf
import matplotlib.pyplot as plt


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(320, activation=tf.nn.tanh),#tf.nn.relu
        tf.keras.layers.Dense(240, activation=tf.nn.sigmoid),  # tf.nn.relu
        tf.keras.layers.Dense(120, activation=tf.nn.relu),  # tf.nn.relu
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ]
)

model.compile(optimizer=tf.keras.optimizers.SGD(0.01),#GradientDescentOptimizer
              loss = tf.keras.losses.sparse_categorical_crossentropy, #tf.kearas.losses.mean_absolute_error
              metrics=['accuracy']
              )

mnist =tf.keras.datasets.mnist

(train_inputs, train_labels), (test_inputs, test_labels)  =mnist.load_data()
train_inputs, test_inputs = train_inputs/255, test_inputs/255
train_inputs=train_inputs.reshape(60000,28,28,1)
test_inputs=test_inputs.reshape(10000,28,28,1)

history = model.fit(train_inputs, train_labels,
                    epochs=3, batch_size=64, verbose=1,
                    validation_data=(test_inputs, test_labels)
                   )
test_loss, test_acc = model.evaluate(test_inputs, test_labels)
print('Test acc:',test_acc)
print('Test loss:', test_loss)
fig = plt.figure()
plt.plot([*range(len(history.history['loss']))], history.history['loss'], color='blue')
plt.plot([*range(len(history.history['val_loss']))], history.history['val_loss'], color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('sparse categorical crossentropy loss')
fig.show()