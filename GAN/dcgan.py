import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.layers import Reshape, BatchNormalization
from keras.optimizers import SGD
from keras.datasets import mnist
from PIL import Image


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


def train(BATCH_SIZE):
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = (X_train.astype(np.float32) - 127.5)/127.5
	X_train = X_train[:, :, :, None]
	X_test = X_test[:, :, :, None]
	# X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
	D = build_discriminator()
	G = build_generator()
	D_on_G = generator_containing_discriminator(G, D)
	d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
	g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
	G.compile(loss='binary_crossentropy', optimizer="SGD")
	D_on_G.compile(loss='binary_crossentropy', optimizer=g_optim)
	D.trainable = True
	D.compile(loss='binary_crossentropy', optimizer=d_optim)
	for epoch in range(100):
		print("Epoch is", epoch)
		print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
		for index in range(int(X_train.shape[0]/BATCH_SIZE)):
			noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
			image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
			generated_images = G.predict(noise, verbose=0)
			X = np.concatenate((image_batch, generated_images))
			y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
			d_loss = D.train_on_batch(X, y)
			print("batch %d d_loss : %f" % (index, d_loss))
			noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
			D.trainable = False
			g_loss = D_on_G.train_on_batch(noise, [1] * BATCH_SIZE)
			D.trainable = True
			print("batch %d g_loss : %f" % (index, g_loss))
			if index % 10 == 9:
				G.save_weights('generator.hdf5')
				D.save_weights('discriminator.hdf5')


if __name__ == "__main__":
	train(BATCH_SIZE=24)