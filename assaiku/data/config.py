import os

from pydantic import BaseModel

from .constants import (
    CATEGORICAL_COLS,
    COLUMNS,
    LABEL_COL,
    NUMERICAL_COLS,
    WEIGHT_COL,
)
from .constants.categories import INCOME


class DataConfig(BaseModel):
    # File paths
    train_data_path: str = "../data/raw/census_income_learn.csv"
    test_data_path: str = "../data/raw/census_income_test.csv"
    train_data_out: str = "../data/processed/train.parquet"
    test_data_out: str = "../data/processed/test.parquet"

    exploration_path: str = "../results/exploration"

    # Data columns
    numerical_cols: list[str] = NUMERICAL_COLS
    categorical_cols: list[str] = CATEGORICAL_COLS
    label: str = LABEL_COL
    label_values: list[str] = INCOME
    weight_col: str = WEIGHT_COL
    columns: list[str] = COLUMNS

    # Parameters
    perform_exploration: bool = True
    remove_duplicates: bool = True
    threshold_outlier: float | None = 18.0
    drop_age: bool = True

    @property
    def features(self) -> list[str]:
        return self.numerical_cols + self.categorical_cols

    @property
    def mixed_categorical_cols(self) -> list[str]:
        return [*self.categorical_cols, self.label]

    @property
    def features_labels(self) -> list[str]:
        return [*self.features, self.label]

    def path_in_explo(self, filename: str) -> str:
        return os.path.join(self.exploration_path, filename)
