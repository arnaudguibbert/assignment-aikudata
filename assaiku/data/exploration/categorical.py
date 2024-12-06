import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from assaiku.data.config import DataConfig
from assaiku.utils import create_folders
from scipy.stats import chi2_contingency

def proportion_comp(data: pd.DataFrame) -> pd.DataFrame:
    data["proportion"] = data["count"] / data["count"].sum()
    return data

def compute_contigency_tab(data: pd.DataFrame, cat_col: str, label_col: str, weight_col: str) -> pd.DataFrame:
    
    cross_tab = pd.crosstab(
        data[cat_col], data[label_col], values=data[weight_col], aggfunc="sum"
    ).reset_index()

    return cross_tab

def compute_dist(
    data: pd.DataFrame, cat_col: str, label_col: str, weight_col: str
) -> pd.DataFrame:
    cross_tab = compute_contigency_tab(data=data,
                                       cat_col=cat_col,
                                       label_col=label_col,
                                       weight_col=weight_col)

    melt_df = pd.melt(cross_tab, id_vars=[cat_col], value_name="count")

    return (
        melt_df.groupby([label_col])
        .apply(proportion_comp, include_groups=False)
        .reset_index()
    )


def visualize_independance(data: pd.DataFrame,
    data_config: DataConfig,
    folder_path: str | None = None,
    feat_col: str = "feature",
    p_val_col: str = "p value") -> None:

    if folder_path is not None:
        create_folders(folder_path)

    categorical_cols = data_config.categorical_cols
    label_col, weight_col = data_config.label, data_config.weight_col

    p_values = []

    for col in categorical_cols:

        observed = compute_contigency_tab(data=data, cat_col=col, label_col=label_col, weight_col=weight_col)

        # print(observed.iloc[:,1:].to_markdown())

        res = chi2_contingency(observed=observed.iloc[:,1:].values)

        print(res)

        p_values.append({feat_col: col, p_val_col: res.pvalue})

    p_values_data = pd.DataFrame.from_records(p_values)
    p_values_data.sort_values(by=p_val_col,inplace=True)

    print(p_values_data.to_markdown())

    fig = plt.figure(figsize=(12,7))
    ax = sns.barplot(data=p_values_data,x=feat_col,y=p_val_col)
    ax.tick_params(axis="x", rotation=20)
    ax.set_yscale("log")
    ax.set_title("Chi2 test p-values [categorical feature X income label]")

    if folder_path is not None:
        file_path = os.path.join(folder_path,f"chi_squared.png")
        # print("Saving figure")
        # logger.info("Saving figure for categorical feature %s", col)
        fig.savefig(file_path)



def visualize_categorical_dist(
    data: pd.DataFrame,
    data_config: DataConfig,
    folder_path: str | None = None,
    filter_cols: list[str] | None = None,
) -> None:
    
    if folder_path is not None:
        create_folders(folder_path)

    categorical_cols = data_config.categorical_cols
    label_col, weight_col = data_config.label, data_config.weight_col

    for col in categorical_cols:
        # print(f"Analysing {col} categorical feature")

        if filter_cols is not None:
            if col not in filter_cols:
                continue

        fig = plt.figure(figsize=(12,7))
        dist = compute_dist(
            data=data, cat_col=col, label_col=label_col, weight_col=weight_col
        )

        ax = sns.barplot(data=dist, x=col, y="proportion", hue=label_col)
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=60)
        ax.set_title(f"Distribution of {col} in the population")

        fig.tight_layout()

        if folder_path is not None:
            file_path = os.path.join(folder_path,f"cat_{col}")
            # print("Saving figure")
            # logger.info("Saving figure for categorical feature %s", col)
            fig.savefig(file_path)
