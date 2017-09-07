import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import Reshape, BatchNormalization
from keras.optimizers import SGD
from PIL import Image


image_size = (52, 52)


def build_generator():
	model = Sequential()
	model.add(Flatten(input_shape=(52, 52, 3)))
	model.add(Dense(1024))
	model.add(Dense(128*13*13))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Reshape((13, 13, 128)))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(64, (5, 5), padding='same'))
	model.add(Activation('tanh'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(3, (5, 5), padding='same'))
	model.add(Activation('sigmoid'))
	return model


def build_discriminator():
	model = Sequential()
	model.add(Conv2D(64, (5, 5), padding='same', input_shape=(52, 52, 3)))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (5, 5)))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('tanh'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	return model


def build_GAN(G, D):
	model = Sequential()
	model.add(G)
	# 判別器を判別に使うために学習は止める
	D.trainable = False
	model.add(D)
	return model


def load_data():
	x_train = np.array([])
	y_train = np.array([])
	counter = 0
	for file in os.listdir('images/'):
		# 拡張子が.jpgでなければ
		if os.path.splitext(file)[1] != '.jpg':
			continue
		image = Image.open('images/' + file)
		image = image.resize(image_size)
		x_train = np.append(x_train, image)
		counter = counter + 1
	for file in os.listdir('converted/'):
		# 拡張子が.jpgでなければ
		if os.path.splitext(file)[1] != '.jpg':
			continue
		image = Image.open('converted/' + file)
		image = image.resize(image_size)
		y_train = np.append(y_train, image)
		# counter = counter + 1
	# 正規化する必要あり
	x_train = x_train / 255
	y_train = y_train / 255
	x_train = x_train.reshape(counter, image_size[0], image_size[1], 3)
	y_train = y_train.reshape(counter, image_size[0], image_size[1], 3)
	return (x_train, y_train)


def train(epochs, batch_size):
	(x_train, y_train) = load_data()
	sgd1 = SGD(lr=0.0005, momentum=0.9, nesterov=True)
	sgd2 = SGD(lr=0.002, momentum=0.9, nesterov=True)
	G = build_generator()
	# ↓これは要らないのではないか
	# G.compile(loss='binary_crossentropy', optimizer="SGD")
	D = build_discriminator()
	D.compile(loss='binary_crossentropy', optimizer=sgd1)
	GAN = build_GAN(G, D)
	GAN.compile(loss='binary_crossentropy', optimizer=sgd2)
	json_data = G.to_json()
	open('G_model.json', 'w').write(json_data)
	json_data = D.to_json()
	open('D_model.json', 'w').write(json_data)
	for epoch in range(epochs):
		# トレーニングデータを順に与える
		index = (epochs * batch_size) % len(x_train)
		real_images = x_train[index:index+batch_size]
		real_convert = y_train[index:index+batch_size]
		gen_convert = G.predict(real_images)
		real_pair = np.concatenate((real_images, real_convert))
		fake_pair = np.concatenate((real_images, gen_convert))
		pairs = np.concatenate((real_pair, fake_pair))
		print(pairs.shape)
		answer = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))
		D_loss = D.train_on_batch(pairs, answer)
		answer = np.ones(batch_size)
		G_loss = GAN.train_on_batch(real_images, answer)
		print('Epoch ' + str(epoch) + '/' + str(epochs))
		print('G loss: ' + str(G_loss) + ' - D loss: ' + str(D_loss))
		if epoch % 100 == 0:
			G.save_weights('G_weights.hdf5')
			D.save_weights('D_weights.hdf5')


if __name__ == '__main__':
	train(epochs=10000, batch_size=24)