import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM


# 実際のところ不必要
cycle_number = 2
sampling_width = 0.1
len_sequence = 10


def gen_values():
	# 正弦波の離散データを生成
	values = np.arange(-len_sequence * sampling_width, 2 * np.pi * cycle_number, sampling_width)
	values = np.sin(values)
	return values


def print_predict():
	json_data = open('model.json').read()
	model = model_from_json(json_data)
	model.load_weights('weights.hdf5')
	print('Enter the number of predict values (integer)')
	value = input('>> ').rstrip()
	print('Start predicting')
	results = np.array([])
	x_test = gen_values()
	# ミニバッチ作成
	x_test = x_test[0:len_sequence]
	# 次元を合わせる
	x_test = x_test.reshape(1, len_sequence, 1)
	for i in range(int(value)):
		# 予測値から更に未来の状態を予測
		result = model.predict(x_test)
		results = np.append(results, result)
		x_test = np.append(x_test, result)
		x_test = x_test[1:]
		x_test = x_test.reshape(1, len_sequence, 1)
	plt.plot(results)
	plt.show()

if __name__ == '__main__':
	print_predict()