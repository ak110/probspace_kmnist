#!/usr/bin/env python3
"""モデルその1。"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import sklearn.metrics

import pytoolkit as tk

logger = tk.log.get(__name__)

CV_COUNT = 5
SPLIT_SEED = 123
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 128
DATA_DIR = pathlib.Path(f"data")
MODELS_DIR = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
CACHE_DIR = pathlib.Path(f"cache/{MODELS_DIR.name}")


def _main():
    tk.utils.better_exceptions()
    base_parser = argparse.ArgumentParser()
    subparsers = base_parser.add_subparsers(dest="mode")
    subparsers.required = True

    parser = subparsers.add_parser("check")
    parser.set_defaults(handler=_check)

    parser = subparsers.add_parser("train")
    parser.set_defaults(handler=_train)
    parser.add_argument("--cv-index", default=0, type=int)

    parser = subparsers.add_parser("validate")
    parser.set_defaults(handler=_validate)
    parser.add_argument("--cv", action="store_true")

    parser = subparsers.add_parser("predict")
    parser.set_defaults(handler=_predict)
    parser.add_argument("--cv", action="store_true")

    args = base_parser.parse_args()
    args.handler(args)


@tk.log.trace()
@tk.dl.wrap_session()
def _check(args):
    tk.log.init(None)
    model = create_model()
    tk.training.check(model, plot_path=MODELS_DIR / "model.svg")


@tk.log.trace()
@tk.dl.wrap_session(use_horovod=True)
def _train(args):
    tk.log.init(MODELS_DIR / f"{args.mode}.fold{args.cv_index}.log")
    (X_train, y_train), (X_val, y_val) = _load_data(args.cv_index)
    train_dataset = MyDataset(X_train, y_train, INPUT_SHAPE, data_augmentation=True)
    val_dataset = MyDataset(X_val, y_val, INPUT_SHAPE)

    model = create_model()
    callbacks = [
        tk.callbacks.CosineAnnealing(),
        tk.callbacks.Checkpoint(MODELS_DIR / f"model.fold{args.cv_index}.h5", 7),
    ]
    tk.training.train(
        model,
        train_dataset,
        val_dataset,
        batch_size=BATCH_SIZE,
        epochs=300,
        callbacks=callbacks,
        # validation_freq=0,
        model_path=MODELS_DIR / f"model.fold{args.cv_index}.h5",
        workers=8,
        use_multiprocessing=True,
        data_parallel=False,
    )


@tk.log.trace()
@tk.dl.wrap_session()
def _validate(args):
    tk.log.init(MODELS_DIR / f"{args.mode}.log")
    _, y = _load_train_data()
    # 予測
    proba = predict_oof(force_rerun=True)
    pred = proba.argmax(axis=-1)
    # スコア表示
    acc = sklearn.metrics.accuracy_score(y, pred)
    logger.info(f"Accuracy: {acc:.4f}")


@tk.log.trace()
@tk.dl.wrap_session()
def _predict(args):
    tk.log.init(MODELS_DIR / f"{args.mode}.log")
    # 予測
    proba_list = predict_test(force_rerun=True)
    pred = np.mean(proba_list, axis=0).argmax(axis=-1)
    # 保存
    df = pd.DataFrame()
    df["ImageId"] = np.arange(1, len(pred) + 1)
    df["Label"] = pred
    df.to_csv(MODELS_DIR / "submit.csv", index=False)


@tk.cache.memorize(CACHE_DIR)
@tk.dl.wrap_session()
def predict_oof():
    """訓練データを予測して結果を返す。"""
    X, y = _load_train_data()
    train_dataset = MyDataset(X, y, INPUT_SHAPE)
    gpus = tk.dl.get_gpu_count()
    return tk.training.predict_cv(
        CV_COUNT,
        train_dataset,
        oof=True,
        folds=tk.ml.get_folds(X, y, CV_COUNT, SPLIT_SEED, stratify=False),
        batch_size=BATCH_SIZE * gpus,
        models_dir=MODELS_DIR,
        on_batch_fn=_tta,
        load_model_fn=lambda p: tk.models.load(p, gpus=gpus),
    )


@tk.cache.memorize(CACHE_DIR)
@tk.dl.wrap_session()
def predict_test():
    """テストデータを予測して結果を返す。5fold分。"""
    X_test = _load_test_data()
    test_dataset = MyDataset(X_test, np.zeros((len(X_test),)), INPUT_SHAPE)
    gpus = tk.dl.get_gpu_count()
    return tk.training.predict_cv(
        CV_COUNT,
        test_dataset,
        batch_size=BATCH_SIZE * gpus,
        models_dir=MODELS_DIR,
        on_batch_fn=_tta,
        load_model_fn=lambda p: tk.models.load(p, gpus=gpus),
    )


def _tta(model, X_batch):
    """TTAありの予測。"""
    import itertools

    X_batch_pat = np.pad(X_batch, ((0, 0), (2, 2), (2, 2), (0, 0)), "edge")
    crop_patterns = itertools.product([0, 2, 4], [0, 2, 4])
    proba_list = []
    for crop_x, crop_y in crop_patterns:
        proba = model.predict_on_batch(
            X_batch_pat[:, crop_y : crop_y + 28, crop_x : crop_x + 28]
        )
        proba_list.append(proba)
    return np.mean(proba_list, axis=0)


def _load_data(cv_index):
    """データの読み込み。"""
    X, y = _load_train_data()
    train_indices, val_indices = tk.ml.cv_indices(
        X, y, CV_COUNT, cv_index, SPLIT_SEED, stratify=False
    )
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    return (X_train, y_train), (X_val, y_val)


def _load_train_data():
    X = np.load(DATA_DIR / "kmnist-train-imgs.npz")["arr_0"]
    y = np.load(DATA_DIR / "kmnist-train-labels.npz")["arr_0"]
    return X, y


def _load_test_data():
    X_test = np.load(DATA_DIR / "kmnist-test-imgs.npz")["arr_0"]
    return X_test


def create_model():
    """モデルの作成。"""
    inputs = x = tk.keras.layers.Input((None, None, 1))
    x = tk.layers.Preprocess(mode="tf")(x)
    x = _conv2d(128)(x)  # 28
    x = _blocks(128, 4)(x)
    x = _down(256, use_act=False)(x)  # 14
    x = _blocks(256, 4)(x)
    x = _down(512, use_act=False)(x)  # 7
    x = _blocks(512, 4)(x)
    x = tk.keras.layers.GlobalAveragePooling2D()(x)
    x = tk.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        kernel_regularizer=tk.keras.regularizers.l2(1e-4),
    )(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    base_lr = 1e-3 * BATCH_SIZE * tk.hvd.size()
    optimizer = tk.keras.optimizers.SGD(lr=base_lr, momentum=0.9, nesterov=True)
    tk.models.compile(model, optimizer, "categorical_crossentropy", ["acc"])
    return model


def _down(filters, use_act=True):
    def layers(x):
        g = tk.keras.layers.Conv2D(
            1,
            3,
            padding="same",
            activation="sigmoid",
            kernel_regularizer=tk.keras.regularizers.l2(1e-4),
        )(x)
        x = tk.keras.layers.multiply([x, g])
        x = _conv2d(filters, strides=2, use_act=use_act)(x)
        return x

    return layers


def _blocks(filters, count):
    def layers(x):
        for _ in range(count):
            sc = x
            x = _conv2d(filters, use_act=True)(x)
            x = _conv2d(filters, use_act=False)(x)
            x = tk.keras.layers.add([sc, x])
        x = _bn_act()(x)
        return x

    return layers


def _conv2d(filters, kernel_size=3, strides=1, use_act=True):
    def layers(x):
        x = tk.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tk.keras.regularizers.l2(1e-4),
        )(x)
        x = _bn_act(use_act=use_act)(x)
        return x

    return layers


def _bn_act(use_act=True):
    def layers(x):
        x = tk.keras.layers.BatchNormalization(
            gamma_regularizer=tk.keras.regularizers.l2(1e-4)
        )(x)
        x = tk.layers.MixFeat()(x)
        x = tk.keras.layers.Activation("relu")(x) if use_act else x
        return x

    return layers


class MyDataset(tk.data.Dataset):
    """Dataset。"""

    def __init__(self, X, y, input_shape, data_augmentation=False):
        self.X = X
        self.y = y
        self.input_shape = input_shape
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug = tk.image.Compose(
                [
                    tk.image.RandomTransform(
                        width=input_shape[1], height=input_shape[0], flip_h=False
                    ),
                    tk.image.RandomCompose(
                        [
                            tk.image.RandomBlur(p=0.125),
                            tk.image.RandomUnsharpMask(p=0.125),
                            tk.image.GaussNoise(p=0.125),
                            tk.image.RandomBrightness(p=0.25),
                            tk.image.RandomContrast(p=0.25),
                            tk.image.RandomEqualize(p=0.0625),
                            tk.image.RandomAutoContrast(p=0.0625),
                            tk.image.RandomPosterize(p=0.0625),
                            tk.image.RandomAlpha(p=0.125),
                        ]
                    ),
                    tk.image.RandomErasing(),
                ]
            )
        else:
            self.aug = tk.image.Resize(width=input_shape[1], height=input_shape[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.data_augmentation:
            sample1 = self.get_sample(index)
            sample2 = self.get_sample(np.random.choice(len(self)))
            X, y = tk.ndimage.mixup(sample1, sample2)
        else:
            X, y = self.get_sample(index)
        return X, y

    def get_sample(self, index):
        X = self.X[index].reshape((28, 28, 1))
        X = self.aug(image=X)["image"]
        y = tk.keras.utils.to_categorical(self.y[index], NUM_CLASSES)
        return X, y


if __name__ == "__main__":
    _main()
