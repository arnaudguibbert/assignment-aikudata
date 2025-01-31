from pathlib import Path

from assaiku.data import DataConfig
from assaiku.utils import create_folders

from .exploration import (
    analyze_label_dist,
    analyze_nans,
    visualize_categorical_dist,
    visualize_continuous_dist,
    visualize_correlation,
    visualize_distance,
)
from .processing import filter_on_age, filter_outliers, remove_group_duplicates
from .validation import load_and_validate


class DataPipe:
    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config

    def run(self) -> None:
        # Loading and validating data
        train_data, test_data = load_and_validate(data_config=self.data_config)

        if self.data_config.remove_duplicates:
            # removing duplicates and grouping instances
            print("Removing duplicates train set")
            train_data = remove_group_duplicates(
                data=train_data,
                weight_col=self.data_config.weight_col,
            )
            print("Removing duplicates test set")
            test_data = remove_group_duplicates(
                data=test_data,
                weight_col=self.data_config.weight_col,
            )

        # Perform data exploration
        if self.data_config.perform_exploration:
            train_data_explo, test_data_explo = (
                train_data.copy(),
                test_data.copy(),
            )

            analyze_nans(data=train_data_explo)
            analyze_nans(data=test_data_explo)

            analyze_label_dist(train_data_explo, self.data_config)
            analyze_label_dist(test_data_explo, self.data_config)

            train_data_explo = filter_on_age(data=train_data)

            visualize_correlation(
                data=train_data,
                data_config=self.data_config,
                folder_path=self.data_config.path_in_explo("train_continuous"),
            )

            visualize_continuous_dist(
                data=train_data_explo,
                data_config=self.data_config,
                folder_path=self.data_config.path_in_explo("train_continuous"),
            )

            visualize_distance(
                data=train_data,
                data_config=self.data_config,
                folder_path=self.data_config.path_in_explo("train_categorical"),
            )

            visualize_categorical_dist(
                data=train_data_explo,
                data_config=self.data_config,
                folder_path=self.data_config.path_in_explo("train_categorical"),
            )

            del train_data_explo
            del test_data_explo

        else:
            # Filtering outliers
            train_data, test_data = filter_outliers(
                train_data=train_data,
                test_data=test_data,
                numerical_cols=self.data_config.numerical_cols,
                threshold=self.data_config.threshold_outlier,
                folder_path=self.data_config.exploration_path,
            )

            if self.data_config.drop_age:
                train_data = filter_on_age(train_data)
                test_data = filter_on_age(test_data)

            create_folders(
                folder_path=str(Path(self.data_config.train_data_out).parent)
            )
            create_folders(
                folder_path=str(Path(self.data_config.test_data_out).parent)
            )

            # Saving data
            train_data.to_parquet(path=self.data_config.train_data_out)
            test_data.to_parquet(path=self.data_config.test_data_out)
