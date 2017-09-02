import os
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import Reshape, BatchNormalization
from keras.optimizers import SGD
from PIL import Image
import math


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
	images = images.reshape((images.shape[0:3]))
	for i in range(len(images)):
		Image.fromarray(images[i]).save('gen/result' + str(i) + '.png')


def generate(batch_size, nice=False):
	json_data = open('G_model.json').read()
	G = model_from_json(json_data)
	G.load_weights('G_weights.hdf5')
	# G.compile(loss='binary_crossentropy', optimizer="SGD")
	json_data = open('D_model.json').read()
	D = model_from_json(json_data)
	D.load_weights('D_weights.hdf5')
	GAN = build_GAN(G, D)
	noise = np.random.uniform(-1, 1, (batch_size, 100))
	gen_images = G.predict(noise, verbose=1)
	gen_images = gen_images * 255
	save_images(gen_images)


if __name__ == '__main__':
	generate(batch_size=24)