import numpy as np
import math
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adagrad
from keras.optimizers import Adam


class_number = 2
image_size = (100, 100)
input_shape = (image_size[0], image_size[0], 3)
batch_size = 5
epochs = 500
adam = Adam(lr=0.0001)


def build_model():
	# 再帰NNを作成
	model = Sequential()
	model.add(LSTM(10, input_shape=(100, 1)))
	model.add(Dense(1))
	model.add(Activation('linear'))
	return model


def load_data():
	x_train = []
	y_train = []
	for i in range(100):
		x_train.append(math.sin(2 * math.pi * i / 50))
	print(x_train)
	plt.plot(x_train)
	plt.show()
	# return (x_train, y_train)


def main():
	print('loading datas')
	(x_train, y_train) = load_data()
	print('Building a model')
	model = build_model()
	model.compile(loss='mse', optimizer='rmsprop')
	print('Start learning')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
	print('Save model as model.json')
	json_data = model.to_json()
	open('model.json', 'w').write(json_data)
	print('Save weights as weights.hdf5')
	model.save_weights('weights.hdf5')


if __name__ == '__main__':
	load_data()