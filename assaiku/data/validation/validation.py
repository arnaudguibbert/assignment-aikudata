import pandas as pd

from assaiku.data.config import DataConfig
from assaiku.data.load import load_data

from .model import DataValidator


def preprocess_str(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip()


def validate_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_config: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col_name in data_config.mixed_categorical_cols:
        train_data[col_name] = pd.Categorical(
            preprocess_str(train_data[col_name])
        )
        test_data[col_name] = pd.Categorical(
            preprocess_str(test_data[col_name]),
            categories=train_data[col_name].cat.categories,
        )

    val_train_data = DataValidator(train_data)
    val_test_data = DataValidator(test_data)

    return val_train_data, val_test_data


def load_and_validate(
    data_config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = load_data(data_config)
    train_df, test_df = validate_data(
        train_data=train_df, test_data=test_df, data_config=data_config
    )
    return train_df, test_df
