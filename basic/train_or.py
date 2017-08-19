import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [1]])
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

sgd = SGD(lr=0.2, momentum=0.02)
model.compile(loss='mean_squared_error', optimizer=sgd)

print('Start learning OR')
model.fit(x_train, y_train, epochs=2000, batch_size=1)
predict = model.predict(x_test, batch_size=1)
print(predict)

print('Save model as model.json')
json_data = model.to_json()
open('model.json', 'w').write(json_data)
print('Save weights as weight.hdf5')
model.save_weights('weights.hdf5')