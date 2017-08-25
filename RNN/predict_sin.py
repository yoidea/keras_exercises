import pandas as pd
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


def _load_data(data, n_prev = 100):
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY


def train_test_split(df, test_size=0.1, n_prev=100):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)
    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    random.seed(0) # 乱数のシード値
    random_factor = 0.05 # 乱数影響係数
    steps_per_cycle = 80 # ステップ数 / 周期
    number_of_cycles = 2 # 周期数

    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    # df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle) + random.uniform(-1.0, +1.0) * random_factor))
    df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)) + math.sin(x * (8 * math.pi / steps_per_cycle)))
    plt.plot(df["sin_t"][:200])
    plt.show() # 教師データの可視化

    length_of_sequences = 10 # シーケンス長
    (X_train, y_train), (X_test, y_test) = train_test_split(df[["sin_t"]], n_prev=length_of_sequences)
    in_out_neurons = 1 # 入出力サイズ
    hidden_neurons = 10 # 隠れ層サイズ
    # RNNを作ります
    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    exit()
    # 最適化方法を定義
    fitting = model.fit(X_train, y_train, batch_size=300, epochs=30, validation_split=0.05)
    # 予測値を取得
    predicted = model.predict(X_test)
    
    dataf = pd.DataFrame(predicted[:200])
    dataf.columns = ["predict"]
    dataf["input"] = y_test[:200]
    plt.plot(dataf["predict"])
    plt.show() # 予測値の可視化
    plt.plot(dataf["input"])
    plt.show()
    plt.plot(fitting.history['loss'])
    plt.show() # 損失関数の可視化