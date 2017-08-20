import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from PIL import Image
import os


x_train = []
y_train = []
x_test = []

for file in os.listdir('images/'):
	label = [0, 0]
	if file[:3] == 'coc':
		label = [1, 0]
	elif file[:3] == 'bee':
		label = [0, 1]
	image = Image.open('images/' + file).resize((100, 100))
	data = np.array(image)
	data = data.transpose(2, 0, 1)
	print(data)
	print(data.shape)
	x_train.append(data / 255)
	y_train.append([label])

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(3, 100, 100)))
model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
adam = Adam(lr=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

print('Start learning')

# model.fit(x_train, y_train,
#           batch_size=18,
#           epochs=1000,
#           verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=25)

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# predict = model.predict(x_test, batch_size=1)
# print(predict)

# print('Save model as model.json')
# json_data = model.to_json()
# open('model.json', 'w').write(json_data)
# print('Save weights as weight.hdf5')
# model.save_weights('weights.hdf5')