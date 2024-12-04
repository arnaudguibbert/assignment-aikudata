import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from assaiku.data.config import DataConfig


def proportion_comp(data: pd.DataFrame) -> pd.DataFrame:
    data["proportion"] = data["count"] / data["count"].sum()
    return data


def compute_dist(
    data: pd.DataFrame, cat_col: str, label_col: str, weight_col: str
) -> pd.DataFrame:
    cross_tab = pd.crosstab(
        data[cat_col], data[label_col], values=data[weight_col], aggfunc="sum"
    ).reset_index()

    melt_df = pd.melt(cross_tab, id_vars=[cat_col], value_name="count")

    return (
        melt_df.groupby([label_col])
        .apply(proportion_comp, include_groups=False)
        .reset_index()
    )


def visualize_categorical_dist(
    data: pd.DataFrame,
    data_config: DataConfig,
    n_cols: int = 3,
    save_path: str | None = None,
) -> None:
    categorical_cols = data_config.categorical_cols
    n_rows = math.ceil(len(categorical_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 10))
    label_col, weight_col = data_config.label, data_config.weight_col

    for col, ax in zip(categorical_cols, axes.flatten(), strict=False):
        print(f"Analysing {col} categorical feature")

        dist = compute_dist(
            data=data, cat_col=col, label_col=label_col, weight_col=weight_col
        )

        sns.barplot(data=dist, x=col, y="proportion", hue=label_col, ax=ax)
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=60)

    fig.tight_layout()

    if save_path is not None:
        print("Saving figure")
        fig.savefig(save_path)
