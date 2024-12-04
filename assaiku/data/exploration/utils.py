import pandas as pd

from assaiku.data.config import DataConfig


def analyze_nans(data: pd.DataFrame) -> None:
    n_nans = pd.isna(data).sum().sum()
    print("Number of nans in the data:", n_nans)


def analyze_label_dist(data: pd.DataFrame, data_config: DataConfig) -> None:
    count = (
        data[[data_config.label, data_config.weight_col]]
        .groupby(data_config.label, observed=True)
        .sum()
        .reset_index()
    )
    count.rename(columns={data_config.weight_col: "proportion"}, inplace=True)
    count["proportion"] /= count["proportion"].sum()
    print("Distribution of labels")
    print(count.to_markdown())
