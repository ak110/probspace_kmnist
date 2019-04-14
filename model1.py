#!/usr/bin/env python3
"""実験用軽量版モデル。"""
import argparse
import pathlib

import albumentations as A
import numpy as np
import pandas as pd

import pytoolkit as tk

logger = tk.log.get(__name__)

CV_COUNT = 5
SPLIT_SEED = 123
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def _main():
    tk.utils.better_exceptions()
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument('--data-dir', default=pathlib.Path(f'data'), type=pathlib.Path)
    base_parser.add_argument('--models-dir', default=pathlib.Path(f'models/{pathlib.Path(__file__).stem}'), type=pathlib.Path)
    subparsers = base_parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser = subparsers.add_parser('check')
    parser.set_defaults(handler=_check)

    parser = subparsers.add_parser('train')
    parser.set_defaults(handler=_train)
    parser.add_argument('--cv-index', default=0, type=int)

    parser = subparsers.add_parser('validate')
    parser.set_defaults(handler=_validate)
    parser.add_argument('--cv', action='store_true')

    parser = subparsers.add_parser('predict')
    parser.set_defaults(handler=_predict)
    parser.add_argument('--cv', action='store_true')

    args = base_parser.parse_args()
    args.handler(args)


@tk.log.trace()
@tk.dl.wrap_session()
def _check(args):
    _ = args
    tk.log.init(None)
    model = _create_network()
    model.summary()


@tk.log.trace()
@tk.dl.wrap_session(use_horovod=True)
def _train(args):
    tk.log.init(args.models_dir / f'{args.mode}.fold{args.cv_index}.log')
    epochs = 300
    batch_size = 128
    base_lr = 1e-3 * batch_size * tk.hvd.get().size()

    (X_train, y_train), (X_val, y_val) = _load_data(args.data_dir, args.cv_index)
    train_dataset = MyDataset(args.data_dir, X_train, y_train, INPUT_SHAPE, data_augmentation=True)
    val_dataset = MyDataset(args.data_dir, X_val, y_val, INPUT_SHAPE)
    train_data = tk.data.DataLoader(train_dataset, batch_size, shuffle=True, mixup=True, mp_size=tk.hvd.get().size())
    val_data = tk.data.DataLoader(val_dataset, batch_size * 2, shuffle=True, mp_size=tk.hvd.get().size())

    model = _create_network()
    tk.models.load_weights(model, args.models_dir / f'model.fold{args.cv_index}.h5', by_name=True)
    optimizer = tk.optimizers.NSGD(lr=base_lr, momentum=0.9, nesterov=True)
    optimizer = tk.hvd.get().DistributedOptimizer(optimizer, compression=tk.hvd.get().Compression.fp16)
    model.compile(optimizer, 'categorical_crossentropy', ['acc'])

    callbacks = [
        tk.callbacks.CosineAnnealing(),
        tk.hvd.get().callbacks.BroadcastGlobalVariablesCallback(0),
        tk.hvd.get().callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
        tk.callbacks.Checkpoint(args.models_dir / f'model.fold{args.cv_index}.h5', 7),
        tk.callbacks.EpochLogger(),
        tk.callbacks.ErrorOnNaN(),
    ]
    model.fit_generator(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks,
                        verbose=1 if tk.hvd.is_master() else 0)
    tk.models.save(model, args.models_dir / f'model.fold{args.cv_index}.h5')

    if tk.hvd.is_master():
        val_data = tk.data.DataLoader(val_dataset, batch_size * 2)
        values = model.evaluate_generator(val_data, verbose=1)
        for n, v in zip(model.metrics_names, values):
            logger.info(f'{n:8s}: {v:.3f}')


@tk.log.trace()
@tk.dl.wrap_session()
def _validate(args):
    tk.log.init(args.models_dir / f'{args.mode}.log')

    # モデルの読み込み
    models = [tk.models.load(args.models_dir / f'model.fold{cv_index}.h5')
              for cv_index in tk.utils.trange(CV_COUNT if args.cv else 1, desc='load-models')]

    # データの読み込みと予測
    y_true, y_pred = [], []
    for cv_index in tk.utils.trange(CV_COUNT if args.cv else 1, desc='predict-cv'):
        (_, _), (X_val, y_val) = _load_data(args.data_dir, cv_index)
        val_dataset = MyDataset(args.data_dir, X_val, y_val, INPUT_SHAPE)

        pred = tk.models.predict(models[cv_index], val_dataset, batch_size=128, on_batch_fn=_tta)
        pred = pred.argmax(axis=-1)

        y_true.extend(y_val)
        y_pred.extend(pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # スコア表示
    import sklearn.metrics
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    logger.info(f'Accuracy: {acc:.4f}')


@tk.log.trace()
@tk.dl.wrap_session()
def _predict(args):
    tk.log.init(args.models_dir / f'{args.mode}.log')
    # モデルの読み込み
    models = [tk.models.load(args.models_dir / f'model.fold{cv_index}.h5')
              for cv_index in tk.utils.trange(CV_COUNT if args.cv else 1, desc='load-models')]
    # データの読み込み
    X_test = _load_test_data(args.data_dir)
    test_dataset = MyDataset(args.data_dir, X_test, np.zeros((len(X_test),)), INPUT_SHAPE)
    # 予測
    proba_list = [tk.models.predict(model, test_dataset, batch_size=128, on_batch_fn=_tta)
                  for model in tk.utils.tqdm(models, desc='predict')]
    pred = np.mean(proba_list, axis=0).argmax(axis=-1)
    # 保存
    df = pd.DataFrame()
    df['ImageId'] = np.arange(1, len(pred) + 1)
    df['Label'] = pred
    df.to_csv(args.models_dir / 'submit.csv', index=False)


def _tta(model, X_batch):
    """TTAありの予測。"""
    import itertools
    X_batch_pat = np.pad(X_batch, ((0, 0), (2, 2), (2, 2), (0, 0)), 'edge')
    crop_patterns = itertools.product([0, 2, 4], [0, 2, 4])
    proba_list = []
    for crop_x, crop_y in crop_patterns:
        proba = model.predict_on_batch(X_batch_pat[:, crop_y:crop_y + 28, crop_x:crop_x + 28])
        proba_list.append(proba)
    return np.mean(proba_list, axis=0)


def _load_data(data_dir, cv_index):
    """データの読み込み。"""
    X = np.load(data_dir / 'kmnist-train-imgs.npz')['arr_0']
    y = np.load(data_dir / 'kmnist-train-labels.npz')['arr_0']
    train_indices, val_indices = tk.ml.cv_indices(X, y, CV_COUNT, cv_index, SPLIT_SEED, stratify=False)
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    return (X_train, y_train), (X_val, y_val)


def _load_test_data(data_dir):
    X_test = np.load(data_dir / 'kmnist-test-imgs.npz')['arr_0']
    return X_test


def _create_network():
    """ネットワークを作成して返す。"""
    inputs = x = tk.keras.layers.Input((None, None, 1))
    x = tk.layers.Preprocess(mode='tf')(x)
    x = _conv2d(128)(x)  # 28
    x = _blocks(128, 4)(x)
    x = _down(256, use_act=False)(x)  # 14
    x = _blocks(256, 4)(x)
    x = _down(512, use_act=False)(x)  # 7
    x = _blocks(512, 4)(x)
    x = tk.keras.layers.GlobalAveragePooling2D()(x)
    x = tk.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                              kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    return model


def _down(filters, use_act=True):
    def layers(x):
        g = tk.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
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
        x = tk.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_uniform',
                                   kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
        x = _bn_act(use_act=use_act)(x)
        return x
    return layers


def _bn_act(use_act=True):
    def layers(x):
        x = tk.keras.layers.BatchNormalization(gamma_regularizer=tk.keras.regularizers.l2(1e-4))(x)
        x = tk.layers.MixFeat()(x)
        x = tk.keras.layers.Activation('relu')(x) if use_act else x
        return x
    return layers


class MyDataset(tk.data.Dataset):
    """Dataset。"""

    def __init__(self, data_dir, X, y, input_shape, data_augmentation=False):
        self.data_dir = data_dir
        self.X = X
        self.y = y
        self.input_shape = input_shape
        if data_augmentation:
            self.aug = A.Compose([
                tk.image.RandomTransform(width=input_shape[1], height=input_shape[0], flip_h=False),
                tk.image.RandomCompose([
                    tk.image.RandomBlur(p=0.125),
                    tk.image.RandomUnsharpMask(p=0.125),
                    A.GaussNoise(p=0.125),
                    tk.image.RandomBrightness(p=0.25),
                    tk.image.RandomContrast(p=0.25),
                    tk.image.RandomEqualize(p=0.0625),
                    tk.image.RandomAutoContrast(p=0.0625),
                    tk.image.RandomPosterize(p=0.0625),
                    tk.image.RandomAlpha(p=0.125),
                ]),
                tk.image.RandomErasing(),
            ])
        else:
            self.aug = tk.image.Resize(width=input_shape[1], height=input_shape[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index].reshape((28, 28, 1))
        X = self.aug(image=X)['image']
        y = tk.keras.utils.to_categorical(self.y[index], NUM_CLASSES)
        return X, y


if __name__ == '__main__':
    _main()
