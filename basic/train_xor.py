import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
epochs = 2000
sgd = SGD(lr=1.0, momentum=0.02)


def build_model():
	# 2入力1出力3層パーセプトロンを作成
	model = Sequential()
	model.add(Dense(2, input_dim=2))
	model.add(Activation('sigmoid'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	return model


def main():
	print('Building a model')
	model = build_model()
	model.compile(loss='mse', optimizer=sgd)
	# 学習ループ
	print('Start learning')
	model.fit(x_train, y_train, epochs=epochs)
	# 結果の検証
	print('Start verification')
	predict = model.predict(x_test)
	print(predict)
	# モデルの保存
	print('Save model as model.json')
	json_data = model.to_json()
	open('model.json', 'w').write(json_data)
	print('Save weights as weights.hdf5')
	model.save_weights('weights.hdf5')


if __name__ == '__main__':
	main()