import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from assaiku.data import DataConfig

from .binarizer import StaticLabelBinarizer


def split_transform(
    data: pd.DataFrame,
    pipeline: Pipeline,
    label_binarizer: StaticLabelBinarizer,
    data_config: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, weights = (
        data[data_config.features],
        data[data_config.label],
        data[data_config.weight_col].values,
    )

    x_processed = pipeline.transform(x)
    y_bin = label_binarizer.transform(y)

    return x_processed, y_bin, weights
