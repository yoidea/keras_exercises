import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import Reshape, BatchNormalization
from keras.optimizers import SGD
from keras.datasets import mnist
from PIL import Image
import math


def build_generator():
	model = Sequential()
	model.add(Dense(input_dim=100, output_dim=1024))
	model.add(Activation('tanh'))
	model.add(Dense(128*7*7))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(64, (5, 5), padding='same'))
	model.add(Activation('tanh'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(1, (5, 5), padding='same'))
	model.add(Activation('tanh'))
	return model


def build_discriminator():
	model = Sequential()
	model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
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


def generator_containing_discriminator(G, D):
	model = Sequential()
	model.add(G)
	D.trainable = False
	model.add(D)
	return model


def combine_images(generated_images):
	num = generated_images.shape[0]
	width = int(math.sqrt(num))
	height = int(math.ceil(float(num)/width))
	shape = generated_images.shape[1:3]
	image = np.zeros((height*shape[0], width*shape[1]),
					 dtype=generated_images.dtype)
	for index, img in enumerate(generated_images):
		i = int(index/width)
		j = index % width
		image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
			img[:, :, 0]
	return image


def generate(BATCH_SIZE, nice=False):
	G = build_generator()
	G.compile(loss='binary_crossentropy', optimizer="SGD")
	G.load_weights('generator.hdf5')
	if nice:
		D = build_discriminator()
		D.compile(loss='binary_crossentropy', optimizer="SGD")
		D.load_weights('discriminator.hdf5')
		noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
		generated_images = G.predict(noise, verbose=1)
		d_pret = D.predict(generated_images, verbose=1)
		index = np.arange(0, BATCH_SIZE*20)
		index.resize((BATCH_SIZE*20, 1))
		pre_with_index = list(np.append(d_pret, index, axis=1))
		pre_with_index.sort(key=lambda x: x[0], reverse=True)
		nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
		nice_images = nice_images[:, :, :, None]
		for i in range(BATCH_SIZE):
			idx = int(pre_with_index[i][1])
			nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
		image = combine_images(nice_images)
	else:
		noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
		generated_images = G.predict(noise, verbose=1)
		image = combine_images(generated_images)
	image = image*127.5+127.5
	Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


if __name__ == '__main__':
	generate(BATCH_SIZE=24)