import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyod.models.ecod import ECOD

from assaiku.utils import create_folders


def filter_outliers(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    numerical_cols: list[str],
    threshold: int,
    folder_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if folder_path is not None:
        create_folders(folder_path)

    train_data_num = train_data[numerical_cols]
    test_data_num = test_data[numerical_cols]

    ecod = ECOD()
    ecod.fit(train_data_num)

    # Get scores on training data
    y_train_scores = ecod.decision_function(train_data_num)
    non_outlier_train_mask = y_train_scores <= threshold
    n_train_outliers = (~non_outlier_train_mask).sum()

    print(f"Found {n_train_outliers} outlier in train set")

    if folder_path is not None:
        explain_fig = os.path.join(folder_path, "explain_outlier")
        # Get the most confident outlier and explain it
        biggest_outlier_idx = np.argsort(y_train_scores)[-1]
        fig = plt.figure()
        plt.xticks(rotation=60, fontsize=7)
        plt.yscale("linear")
        ecod.explain_outlier(
            biggest_outlier_idx,
            feature_names=train_data_num.columns,
            file_name=explain_fig,
        )
        # Plot outlier scores histogram
        fig = plt.figure()
        ax = sns.histplot(x=y_train_scores)
        ax.set_yscale("log")
        ax.set_xlabel("Outlier score")
        ax.set_title("Distribution of outlier scores on training set")
        fig.savefig(os.path.join(folder_path, "histogram_outliers"))

    # Get scores on training data
    y_test_scores = ecod.decision_function(test_data_num)
    non_outlier_test_mask = y_test_scores <= threshold
    n_test_outliers = (~non_outlier_test_mask).sum()

    print(f"Found {n_test_outliers} outlier in test set")

    print("Filtering out outliers from train and test data")
    train_data = train_data[non_outlier_train_mask]
    test_data = test_data[non_outlier_test_mask]

    return train_data, test_data
