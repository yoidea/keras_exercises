import os
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import Reshape, BatchNormalization
from keras.optimizers import SGD
from PIL import Image
import math


image_size = (52, 52)


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


def build_GAN(G, D):
	model = Sequential()
	model.add(G)
	# 判別器を判別に使うために学習は止める
	D.trainable = False
	model.add(D)
	return model


def save_images(images):
	if os.path.isdir('gen') == False:
		os.mkdir('gen')
	images = images.astype(np.uint8)
	images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 3))
	for i in range(len(images)):
		Image.fromarray(images[i]).save('gen/result' + str(i) + '.jpg')


def generate(batch_size, nice=False):
	(x_train, y_train) = load_data()
	json_data = open('G_model.json').read()
	G = model_from_json(json_data)
	G.load_weights('G_weights.hdf5')
	# G.compile(loss='binary_crossentropy', optimizer="SGD")
	json_data = open('D_model.json').read()
	D = model_from_json(json_data)
	D.load_weights('D_weights.hdf5')
	GAN = build_GAN(G, D)
	real_images = x_train
	gen_images = G.predict(real_images, verbose=1)
	gen_images = gen_images * 255
	save_images(gen_images)


if __name__ == '__main__':
	generate(batch_size=24)