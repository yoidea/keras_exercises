import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout


def print_predict():
	x_test = np.empty((1, 2))
	json_data = open('model.json').read()
	model = model_from_json(json_data)
	model.load_weights('weights.hdf5')
	print('Enter 2 boolean values (0 or 1)')
	while True:
		values = input('>> ').rstrip().split(' ')
		x_test[0][0] = int(values[0])
		x_test[0][1] = int(values[1])
		predict = model.predict(x_test)
		print(predict[0][0])


if __name__ == '__main__':
	print_predict()