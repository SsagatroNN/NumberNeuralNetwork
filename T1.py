import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# xtest = tf.keras.utils.normalize(x_test, axis = 1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

image_number = 0
while os.path.isfile(f'tests/test{image_number}.jpg'):
	img = cv2.imread(f"tests/test{image_number}.jpg")[:,:,0]
	img = np.invert(np.array([img]))
	prediction = model.predict(img)
	print(f"The number is probably {np.argmax(prediction)}")
	plt.imshow(img[0], cmap = plt.cm.binary)
	plt.show()
	image_number += 1
