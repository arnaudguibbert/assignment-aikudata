import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from assaiku.data.config import DataConfig

LOG_THRESHOLD = 2


def visualize_continuous_dist(
    data: pd.DataFrame,
    data_config: DataConfig,
    n_cols: int = 3,
    save_path: str | None = None,
) -> None:
    data = data.copy()

    numerical_cols = data_config.numerical_cols
    n_rows = math.ceil(len(numerical_cols) / n_cols)
    label_col, weight_col = data_config.label, data_config.weight_col

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 5))

    for col, ax in zip(numerical_cols, axes.flatten(), strict=False):
        print(f"Analyzing {col} continuous feature")

        x_log_scale = False
        serie = data[col]
        min_serie = serie.min()

        if min_serie >= 0:
            max_serie = serie.max()
            if math.log10(max_serie / (min_serie + 1)) > LOG_THRESHOLD:
                x_log_scale = True

        sns.histplot(
            data=data,
            x=col,
            hue=label_col,
            weights=weight_col,
            ax=ax,
            stat="proportion",
            bins=min(serie.nunique(), 100),
            common_norm=False,
        )
        if x_log_scale:
            ax.set_xscale("log")
        ax.set_yscale("log")
    fig.tight_layout()

    if save_path is not None:
        print("Saving figure")
        fig.savefig(save_path)
