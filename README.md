# keras_exercises
深層学習を勉強するために作成した、基礎的なニューラルネットワークのプログラム。
可能な限りコードが短くなるようにしたつもりだ。

## 動作環境
以下の環境で動作を確認した。
- Python 3.6.1
- numpy (1.12.1)
- tensorflow (1.1.0)
- Keras (2.0.5)
また、以下のライブラリを利用している。
- matplotlib (2.0.2) : グラフ描画
- h5py (2.7.0) : モデル保存
- Pillow (4.2.1) : 画像処理

## 構成
- `basic/` : 回帰問題を解くニューラルネットワークのプログラム
- `CNN/` : 畳み込みニューラルネットワークを利用したプログラム
- `RNN/` : 再帰的ニューラルネットワークを利用したプログラム
- `GAN/` : 敵対生成ネットワークを利用したプログラム

## 実行
### basic
AND, OR, XORの分類問題を解くことができる。
1. `train_and.py`、`train_or.py`、`train_xor.py`のいずれかを実行する。
1. 学習が完了したら`predict.py`実行する。
1. 真偽値を2値与えると出力を返す。
```bash
python train_xor.py
...
python predict.py
Using TensorFlow backend.
Enter 2 boolean values (0 or 1)
>> 1 0
0.950542
```

### CNN
ゴキブリとカブトムシを判別することができる。
1. `images/`の中に`coc*.jpg`という名前でゴキブリの画像を数十枚用意する。（画像サイズは任意）
1. `images/`の中に`bee*.jpg`という名前でカブトムシの画像を数十枚用意する。（画像サイズは任意）
1. `coc_or_bee.py`を実行する。
1. 学習が完了したら`predict.py`実行する。
1. テスト画像のパスを入力すると判定できる。
```bash
ls images/
bee0.jpg   bee1.jpg  ...
coc0.jpg   coc1.jpg  ...
coc99.jpg
python coc_or_bee.py
...
python predict.py
Using TensorFlow backend.
>> tests/coc_test1.jpg
[ 0.99894518  0.00105484]
ゴキブリ
```

### RNN
正弦波を予測して無限に波形を生成する。
1. まず、`sin.py`を実行する。（突然グラフを描画するが驚かないように注意）
1. 学習が完了したら`predict.py`実行する。
1. 生成する波形のサンプル数を与えると予測値からグラフを描画する。
```bash
python sin.py
...
python predict.py
Enter the number of predict values (integer)
>> 1000
Start predicting
```
![RNN input](img/RNNin)
![RNN output](img/RNNout)

### GAN
mnistのデータを学習して、類似データを生成する。
1. `train.py`を実行する。（mnistデータは自動ダウンロード）
1. 学習が完了したら`predict.py`を実行する。
1. データは`gen/result*.png`として生成される。
```bash
python train.py
...
python predict.py
...
ls gen/
result0.png   result1.png  result2.png  . . .  result23.png
```
![GAN output](img/GANout)