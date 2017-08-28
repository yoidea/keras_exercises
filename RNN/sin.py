import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM


batch_size = 10
epochs = 50
cycle_number = 2
sampling_width = 0.1
len_sequence = 10


def build_model():
	# 再帰NNを作成
	model = Sequential()
	model.add(LSTM(10, input_shape=(10, 1)))
	model.add(Dense(1))
	model.add(Activation('linear'))
	return model


def gen_values():
	# 正弦波の離散データを生成
	values = np.arange(0, 2 * np.pi * cycle_number, sampling_width)
	values = np.sin(values)
	return values


def load_data():
	values = gen_values()
	# (n, 1)に変換
	y_train = values.reshape(len(values), 1)
	# 頭出し
	y_train = y_train[len_sequence:]
	x_train = np.array([])
	# 始点をずらしながらミニバッチを生成
	for i in range(len(values) - len_sequence):
		x_train = np.append(x_train, values[i:i+len_sequence])
	# (n, len_sequence, 1)に変換
	x_train = x_train.reshape(len(values) - len_sequence, len_sequence, 1)
	return (x_train, y_train)


def main():
	print('Loading datas')
	(x_train, y_train) = load_data()
	print('Show training datas')
	plt.plot(y_train)
	plt.show()
	print('Building a model')
	model = build_model()
	model.compile(loss='mse', optimizer='rmsprop')
	print('Start learning')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
	result = model.predict(x_train)
	print('Show results')
	plt.plot(result)
	plt.show()
	print('Save model as model.json')
	json_data = model.to_json()
	open('model.json', 'w').write(json_data)
	print('Save weights as weights.hdf5')
	model.save_weights('weights.hdf5')


if __name__ == '__main__':
	main()