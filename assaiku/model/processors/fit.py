import pandas as pd
from sklearn.pipeline import Pipeline


def fit_processor(
    train_data: pd.DataFrame, feature_cols: list[str], pipeline: Pipeline
) -> None:
    pipeline.fit(train_data[feature_cols])
