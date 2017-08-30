import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categorical
from PIL import Image
import os


input_dim = 10
batch_size = 5
epochs = 500


def generator():
	# Generatorを構築
	model = Sequential()
	model.add(Dense(256, input_dim=100))
	model.add(LeakyReLU(0.2))
	model.add(Dense(512))
	model.add(LeakyReLU(0.2))
	model.add(Dense(1024))
	model.add(Activation('sigmoid'))
	# 不要かもしれない
	sgd = SGD(lr=0.1, momentum=0.3, decay=1e-5)
	model.compile(loss="binary_crossentropy", optimizer=sgd)
	return model


def discriminator():
	# Discriminatorを構築
	model = Sequential()
	model.add(Dense(1024, input_dim=10))
	model.add(LeakyReLU(0.2))
	model.add(Dense(512))
	model.add(LeakyReLU(0.2))
	model.add(Dense(256))
	model.add(LeakyReLU(0.2))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	sgd = SGD(lr=0.1, momentum=0.1, decay=1e-5)
	model.compile(loss="binary_crossentropy", optimizer=sgd)
	return model


def build_GAN():
	# Generative Adversarial Network
	G = generator()
	D = discriminator()
	model = Sequential()
	model.add(G)
	D.model.trainable = False
	model.add(D)
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
	model = build_GAN()
	sgd = SGD(0.1, momentum=0.3)
	model.compile(loss="binary_crossentropy", optimizer=sgd)
	print('Start learning')
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
	print('Save model as model.json')
	json_data = model.to_json()
	open('model.json', 'w').write(json_data)
	print('Save weights as weights.hdf5')
	model.save_weights('weights.hdf5')


if __name__ == '__main__':
	main()