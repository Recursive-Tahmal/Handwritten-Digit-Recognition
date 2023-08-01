import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load the data
mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
# unpacks images to x_train/x_test and labels to y_train/y_test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data
x_train = tf.keras.utils.normalize(
    x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(
    x_test, axis=1)  # scales data between 0 and 1

# build the model
model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
# a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))  # Dropout layer with 20% dropout rate
# a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))  # Dropout layer with 20% dropout rate
# our output layer. 10 units for 10 classes. Softmax for probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model with increased epochs
model.fit(x_train, y_train, epochs=10)

# save the model
model.save('digits.model')

# evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy
