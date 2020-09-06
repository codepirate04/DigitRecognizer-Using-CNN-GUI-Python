
# Simple CNN for the MNIST Dataset
import tensorflow as tf
import cv2 as cv
import numpy as np
# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) #default flaot 64 can use .astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# define a simple CNN model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
	# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

model.save('digitreg2.h5')

image= cv.imread('image.png',cv.IMREAD_GRAYSCALE) #here change image.png with your own image
image=cv.resize(image,(28,28))
image=image.reshape(1,28,28,1)
image=image/255
res=model.predict(image)
store=np.copy(res)
store[0,res.argmax()]=0
max2=store.argmax()
print("The  predicted number is:",res.argmax(),",Second Guess:",max2)





































