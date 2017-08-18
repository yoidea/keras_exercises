import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [0], [0], [1]])
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.array([[0], [0], [0], [1]])

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

sgd = SGD(lr=0.2, momentum=0.02)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_train, y_train, epochs=10000, batch_size=1)
predict = model.predict(x_test, batch_size=1)

print(predict)