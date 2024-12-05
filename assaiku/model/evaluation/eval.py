from typing import Literal

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def evaluate_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    model_name: str,
    data_set: Literal["train", "test"],
) -> dict[str, float | str | int]:
    print(f"Evaluating on {data_set} set")

    y_pred = model.predict(x)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true=y, y_pred=y_pred, sample_weight=weights, pos_label=1
    )

    print(
        f"Model = {model_name} - Precision {prec} - Recall {recall} - F1 {f1}"
    )

    perf_0 = {
        "precision": prec[0],
        "recall": recall[0],
        "f1": f1[0],
        "class": 0,
        "model": model_name,
        "set": data_set,
    }
    perf_1 = {
        "precision": prec[1],
        "recall": recall[1],
        "f1": f1[1],
        "class": 1,
        "model": model_name,
        "set": data_set,
    }

    return perf_0, perf_1
