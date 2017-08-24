import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from PIL import Image
import os

batch_size = 25
num_classes = 2
epochs = 100
x_train = []
y_train = []

for file in os.listdir('images/'):
	label = [0, 0]
	if file[:3] == 'coc':
		label = [1, 0]
	elif file[:3] == 'bee':
		label = [0, 1]
	image = Image.open('images/' + file).resize((100, 100))
	data = np.array(image)
	print(data)
	print(data.shape)
	x_train.append(data / 255)
	y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

adam = Adam(lr=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)