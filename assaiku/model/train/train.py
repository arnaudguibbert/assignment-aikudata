import math

import numpy as np

from assaiku.model.configs import LogisticRegressionConfig


def balance_weights(
    y: np.ndarray,
    weights: np.ndarray,
    weight_neg_factor: float,
    weight_pos_factor: float,
) -> np.ndarray:
    n_samples, n_pos = weights.sum(), (y * weights).sum()
    n_neg = n_samples - n_pos
    pos_weight = math.sqrt(n_samples / n_pos) * weight_pos_factor
    neg_weight = math.sqrt(n_samples / n_neg) * weight_neg_factor

    print("Positive weights set to", pos_weight)
    print("Negative weights set to", neg_weight)

    new_weights = weights * y * pos_weight + (1 - y) * weights * neg_weight
    return new_weights


def train_model(
    model_config: LogisticRegressionConfig,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
) -> None:
    print(f"Training model {model_config.name}")

    if model_config.balance_weights:
        weights = balance_weights(
            y=y_train,
            weights=weights,
            weight_neg_factor=model_config.weight_neg_factor,
            weight_pos_factor=model_config.weight_pos_factor,
        )

    model.fit(x_train, y_train, sample_weight=weights)
