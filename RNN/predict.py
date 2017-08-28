import numpy as np
import math
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from PIL import Image


def load_data():
	x_train = []
	y_train = np.array([np.sin(np.arange(0, 4 * math.pi, 0.1))])
	y_train = y_train.transpose()
	for i in range(len(y_train) - 10):
		x_train.append(y_train[i:i+10])
	x_train = np.array(x_train)
	y_train = y_train[10:]
	return (x_train, y_train)


def print_predict():
	point = []
	json_data = open('model.json').read()
	model = model_from_json(json_data)
	model.load_weights('weights.hdf5')
	print('Start predicting')
	(x_train, y_train) = load_data()
	p_data = np.array([x_train[0]])
	p_value = model.predict(p_data)
	point.append(p_value[0])
	for i in range(1, 200):
		p_data = np.append(p_data, p_value)
		print(p_data.shape)
		p_data = np.array([p_data[1:]])
		print(p_data.shape)
		p_data = p_data.transpose()
		print(p_data.shape)
		p_data = np.array([p_data])
		print(p_data.shape)
		p_value = model.predict(p_data)
		point.append(p_value[0])
	plt.plot(point)
	plt.show()

if __name__ == '__main__':
	print_predict()