import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from PIL import Image


def print_predict():
	json_data = open('model.json').read()
	model = model_from_json(json_data)
	model.load_weights('weights.hdf5')
	print('Enter the file name (*.jpg)')
	while True:
		values = input('>> ').rstrip()
		x_test = []
		image = Image.open(values).resize((100, 100))
		x_test.append(np.array(image) / 255)
		x_test = np.array(x_test)
		predict = model.predict(x_test)
		print(predict[0])
		if predict[0][0] > predict[0][1]:
			print('ゴキブリ')
		else:
			print('カブトムシ')


if __name__ == '__main__':
	print_predict()