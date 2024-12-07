from .data import DataConfig, DataPipe
from .model import MLPipe
from .model.configs import (
    EvaluationConfig,
    LinearSVMConfig,
    LogisticRegressionConfig,
    XGBConfig,
)


def get_data_config(explo: bool) -> DataConfig:
    data_config = DataConfig(
        perform_exploration=explo,
        train_data_path="./data/raw/census_income_learn.csv",
        test_data_path="./data/raw/census_income_test.csv",
        train_data_out="./data/processed/train.parquet",
        test_data_out="./data/processed/test.parquet",
        exploration_path="./results/exploration",
    )

    return data_config


def main() -> None:
    # Data Exploration
    data_config = get_data_config(explo=True)
    data_pipeline = DataPipe(data_config=data_config)
    data_pipeline.run()

    # Data cleaning & preprocessing
    data_config = get_data_config(explo=False)
    data_pipeline = DataPipe(data_config=data_config)
    data_pipeline.run()

    # ML pipeline
    eval_config = EvaluationConfig(
        n_repet=1,  # All models we are testing are determinitic in the way they are trained so far
        model_configs=[
            XGBConfig(
                n_estimators=100,
                max_depth=7,
                learning_rate=1e-1,
                dimension_red=None,
            ),
            XGBConfig(
                weight_neg_factor=1, weight_pos_factor=1, dimension_red=50
            ),
            LinearSVMConfig(rbf_gamma=5e-5, C=100),
            LinearSVMConfig(),
            LogisticRegressionConfig(),
            LogisticRegressionConfig(dimension_red=50),
        ],
        folder_out_result="./results/model",
    )

    ml_pipeline = MLPipe(data_config=data_config, evaluation_config=eval_config)

    ml_pipeline.run()
