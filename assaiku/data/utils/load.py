import pandas as pd

from assaiku.data import DataConfig


def load_data(data_config: DataConfig) -> pd.DataFrame:
    train_data = pd.read_csv(
        data_config.train_data_path, names=data_config.columns
    )
    test_data = pd.read_csv(
        data_config.test_data_path, names=data_config.columns
    )

    return train_data, test_data
