import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adagrad
from keras.optimizers import Adam
# from keras.utils.np_utils import to_categorical
from PIL import Image
import os


class_number = 2
image_size = (100, 100)
input_shape = (image_size[0], image_size[0], 3)
batch_size = 5
epochs = 500
adam = Adam(lr=0.0001)


def build_model():
	# 畳み込みNNを作成
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(64, kernel_size=(3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(class_number))
	model.add(Activation('softmax'))
	return model


def load_data():
	x_train = []
	y_train = []
	for file in os.listdir('images/'):
		if file[:3] == 'coc':
			label = [1, 0]
		elif file[:3] == 'bee':
			label = [0, 1]
		else:
			continue
		image = Image.open('images/' + file)
		image = image.resize(image_size)
		# 正規化する必要あり
		x_train.append(np.array(image) / 255)
		y_train.append(label)
	# numpy型に変換
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	return (x_train, y_train)


def main():
	print('loading datas')
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