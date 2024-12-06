import pandas as pd


def remove_group_duplicates(
    data: pd.DataFrame,
    weight_col: str,
    remove_age: bool = False,
) -> pd.DataFrame:
    n_duplicates = len(data) - len(data.drop_duplicates())
    print("Number of duplicates including instances_weights:", n_duplicates)

    data_wo_weight = data.drop(columns=[weight_col])
    n_duplicates_wo_weight = len(data_wo_weight) - len(
        data_wo_weight.drop_duplicates()
    )
    print(
        "Number of duplicates excluding instances_weights:",
        n_duplicates_wo_weight,
    )

    # Drop duplicates with instances
    print(
        "Dropping duplicates including the instances_weight (systematic error)"
    )
    data = data.drop_duplicates()
    # Group same instances and sum their instance weights
    print("Gropuping same instances and their weight instances")
    feature_label_cols = list(data.columns)
    feature_label_cols.remove(weight_col)
    data = data.groupby(feature_label_cols, observed=True).sum().reset_index()

    if remove_age:
        print("Removing age below 16")
        data = data[data["age"] >= 16]

    print("Number of samples after cleaning:", len(data))

    return data
