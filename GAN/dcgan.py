import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adagrad
from keras.optimizers import Adam
# from keras.utils.np_utils import to_categorical
from PIL import Image
import os


image_dim = 100
batch_size = 5
epochs = 500
adam = Adam(lr=0.0001)


def build_gen_model():
	# Generatorを構築
	model = Sequential()
	model.add(Dense(256, input_dim=input_dim))
	model.add(LeakyReLU(0.2))
	model.add(Dense(512))
	model.add(LeakyReLU(0.2))
	model.add(Dense(1024))
	model.add(Activation('sigmoid'))
	return model


def build_dis_model():
	# Generatorを構築
	model = Sequential()
	model.add(Dense(1024, input_dim=input_dim))
	model.add(LeakyReLU(0.2))
	model.add(Dense(512))
	model.add(LeakyReLU(0.2))
	model.add(Dense(256))
	model.add(LeakyReLU(0.2))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	return model


def load_data():
	x_train = []
	y_train = []
	# numpy型に変換
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	return (x_train, y_train)


def main():
	print('Loading datas')
	(x_train, y_train) = load_data()
	print('Building a model')
	model = build_model()
	model.compile(loss='categorical_crossentropy', optimizer=adam)
	print('Start learning')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
	print('Save model as model.json')
	json_data = model.to_json()
	open('model.json', 'w').write(json_data)
	print('Save weights as weights.hdf5')
	model.save_weights('weights.hdf5')


if __name__ == '__main__':
	main()