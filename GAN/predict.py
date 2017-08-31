import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import Reshape, BatchNormalization
from keras.optimizers import SGD
from PIL import Image
import math


def build_generator():
	model = Sequential()
	model.add(Dense(1024, input_dim=100))
	model.add(Activation('tanh'))
	model.add(Dense(128*7*7))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Reshape((7, 7, 128)))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(64, (5, 5), padding='same'))
	model.add(Activation('tanh'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(1, (5, 5), padding='same'))
	model.add(Activation('sigmoid'))
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


def generate(batch_size, nice=False):
	G = build_generator()
	G.compile(loss='binary_crossentropy', optimizer="SGD")
	G.load_weights('generator.hdf5')
	noise = np.random.uniform(-1, 1, (batch_size, 100))
	gen_images = G.predict(noise, verbose=1)
	image = combine_images(gen_images)
	image = image * 255
	Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


if __name__ == '__main__':
	generate(batch_size=24)