from pydantic import BaseModel

from .constants import (
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    LABEL_COL,
    WEIGHT_COL,
    COLUMNS,
)

from .constants.categories import INCOME


class DataConfig(BaseModel):
    train_data_path: str = "../data/raw/census_income_learn.csv"
    test_data_path: str = "../data/raw/census_income_test.csv"

    numerical_cols: list[str] = NUMERICAL_COLS
    categorical_cols: dict[str, list[str]] = CATEGORICAL_COLS
    label: str = LABEL_COL
    label_values: list[str] = INCOME
    weight_col: str = WEIGHT_COL
    columns: list[str] = COLUMNS

    @property
    def features(self) -> list[str]:
        return self.numerical_cols + list(self.categorical_cols.keys())
