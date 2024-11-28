import tensorflow as tf
import matplotlib.pyplot as plt

from examples.Conv1D.read_dataset import read_acc_data_one_hot

train_X, train_Y, test_X, test_Y = read_acc_data_one_hot()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu'))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu'))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print(train_X.shape)
train_X = train_X.reshape(-1,128,1)
model.fit(train_X, train_Y, epochs=30, batch_size=64, verbose=2)
model.summary()

test_X = test_X.reshape(-1,128,1)
accuracy = model.evaluate(test_X, test_Y, verbose=2)
print("test loss, test accuracy:", accuracy)