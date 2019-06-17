# ProbSpace 「くずし字」識別チャレンジ 2位解法

<https://prob.space/competitions/kuzushiji-mnist>

<https://prob.space/competitions/kuzushiji-mnist/discussions/ak1100-Post7feac3a17f875c9e04dd>

## やったこと

ちょっとずつパラメータを変えた3種類のモデル×各5foldでTTA(9 crops)して平均しただけ。

(初submitでは1種類のモデルだったけど結局スコア変わらなかった)

## つかいかた

### ライブラリ

- Pillow-SIMD
- better-exceptions
- horovod
- numba
- numpy
- opencv-python
- scikit-learn
- scipy
- tensorflow-gpu
- tqdm

[怪しいDockerイメージ](https://github.com/ak110/dl_allinone)を使ってたのでpipenvとかは用意してないけどたぶんこのくらい。

Pillow-SIMDはただのPillowでもよかったり、TensorFlowは1.13.0を使ったけどそこそこあたらしい1.x系ならたぶん動くくらいだったり。

### 学習

    ./run.sh model1.py
    ./run.sh model2.py
    ./run.sh model3.py

OpenMPIが無ければそのまま実行してもたぶん動くつもり。

### 推論

    ./averaging.py

