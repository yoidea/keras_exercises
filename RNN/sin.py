import numpy as np
import math
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM


batch_size = 5
epochs = 50


def build_model():
	# 再帰NNを作成
	model = Sequential()
	model.add(LSTM(10, input_shape=(10, 1)))
	model.add(Dense(1))
	model.add(Activation('linear'))
	return model


def load_data():
	x_train = []
	y_train = np.array([np.sin(np.arange(0, 4 * math.pi, 0.1))])
	y_train = y_train.transpose()
	print(y_train.shape)
	plt.plot(y_train)
	plt.show()
	for i in range(len(y_train) - 10):
		x_train.append(y_train[i:i+10])
	x_train = np.array(x_train)
	print(x_train.shape)
	y_train = y_train[10:]
	print(y_train.shape)
	return (x_train, y_train)


def main():
	print('loading datas')
	(x_train, y_train) = load_data()
	print('Building a model')
	model = build_model()
	model.compile(loss='mse', optimizer='rmsprop')
	print('Start learning')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
	p_value = model.predict(x_train)
	plt.plot(p_value)
	plt.show()
	print('Save model as model.json')
	json_data = model.to_json()
	open('model.json', 'w').write(json_data)
	print('Save weights as weights.hdf5')
	model.save_weights('weights.hdf5')


if __name__ == '__main__':
	main()