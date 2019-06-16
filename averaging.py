#!/usr/bin/env python3
"""加重平均によるアンサンブル。"""
import pathlib

import numpy as np
import pandas as pd

import pytoolkit as tk

logger = tk.log.get(__name__)
MODELS_DIR = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")


def _main():
    tk.utils.better_exceptions()
    tk.log.init(MODELS_DIR / f"predict.log")

    import model1
    import model2
    import model3

    # model1 0.9982
    # model2 0.9980
    # model3 0.9981

    proba = np.average(
        [
            np.mean(model1.predict_test(), axis=0),
            np.mean(model2.predict_test(), axis=0),
            np.mean(model3.predict_test(), axis=0),
        ],
        weights=[3.0, 0.5, 1.0],  # 適当重み
        axis=0,
    )
    pred = proba.argmax(axis=-1)

    # 保存
    df = pd.DataFrame()
    df["ImageId"] = np.arange(1, len(pred) + 1)
    df["Label"] = pred
    df.to_csv(MODELS_DIR / "submit.csv", index=False)


if __name__ == "__main__":
    _main()
