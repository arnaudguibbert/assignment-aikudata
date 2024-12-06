import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from pyod.utils.stat_models import wpearsonr

from assaiku.data.config import DataConfig
from assaiku.utils import create_folders


def visualize_correlation(
    data: pd.DataFrame,
    data_config: DataConfig,
    folder_path: str | None = None,
    feat_col="feature",
    score_col="absolute weighted pearson coeff",
) -> None:
    if folder_path is not None:
        create_folders(folder_path)

    label_col, weight_col = data_config.label, data_config.weight_col
    label_values = data[label_col].cat.codes.astype(float)
    weight_values = data[weight_col].values

    pearson_val = []

    for col in data_config.numerical_cols:
        pearson_corr = wpearsonr(
            x=data[col].values, y=label_values, w=weight_values
        )

        pearson_val.append({feat_col: col, score_col: abs(pearson_corr)})

    pearson_data = pd.DataFrame.from_records(pearson_val)
    pearson_data.sort_values(by=score_col, inplace=True, ascending=False)

    fig = plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=pearson_data, x=feat_col, y=score_col)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(
        "Absoulte correlation coefficient [continous feature X income label]"
    )
    fig.tight_layout()

    if folder_path is not None:
        file_path = os.path.join(folder_path, f"pearson_continuous.png")
        # print("Saving figure")
        # logger.info("Saving figure for categorical feature %s", col)
        fig.savefig(file_path)


def visualize_continuous_dist(
    data: pd.DataFrame,
    data_config: DataConfig,
    folder_path: str | None = None,
    log_threshold: int = 2,
    filter_cols: list[str] | None = None,
) -> None:
    data = data.copy()

    if folder_path is not None:
        create_folders(folder_path)

    numerical_cols = data_config.numerical_cols
    label_col, weight_col = data_config.label, data_config.weight_col

    for col in numerical_cols:
        # print(f"Analyzing {col} continuous feature")

        if filter_cols is not None:
            if col not in filter_cols:
                continue

        fig = plt.figure(figsize=(12, 7))
        x_log_scale = False
        serie = data[col]
        min_serie = serie.min()

        if min_serie >= 0:
            max_serie = serie.max()
            if math.log10(max_serie / (min_serie + 1)) > log_threshold:
                x_log_scale = True

        ax = sns.histplot(
            data=data,
            x=col,
            hue=label_col,
            weights=weight_col,
            stat="proportion",
            bins=min(serie.nunique(), 100),
            common_norm=False,
        )
        if x_log_scale:
            ax.set_xscale("log")

        ax.set_yscale("log")
        ax.set_title(f"Distribution of {col} in the population")
        fig.tight_layout()

        if folder_path is not None:
            file_path = os.path.join(folder_path, f"cont_{col}")
            # print("Saving figure")
            # logger.info("Saving figure for categorical feature %s", col)
            fig.savefig(file_path)
