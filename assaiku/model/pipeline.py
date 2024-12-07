import pandas as pd

from assaiku.data import DataConfig

from .configs import EvaluationConfig
from .evaluation import analyze_data, evaluate_model
from .processors import (
    fit_processor,
    initialize_feat_processor,
    split_transform,
)
from .train import initialize_model, train_model


class MLPipe:
    def __init__(
        self, data_config: DataConfig, evaluation_config: EvaluationConfig
    ) -> None:
        self.data_config = data_config
        self.evaluation_config = evaluation_config

    def run(self) -> None:
        train_data = pd.read_parquet(path=self.data_config.train_data_out)
        test_data = pd.read_parquet(path=self.data_config.test_data_out)

        data = []

        for model_config in self.evaluation_config.model_configs:
            print(f"Fitting processor for model {model_config.name}")

            feat_processor, label_binarizer = initialize_feat_processor(
                data_config=self.data_config, model_config=model_config
            )
            fit_processor(
                train_data=train_data,
                feature_cols=self.data_config.features,
                pipeline=feat_processor,
            )

            x_train, y_train, w_train = split_transform(
                train_data,
                feat_processor,
                label_binarizer,
                data_config=self.data_config,
            )
            x_test, y_test, w_test = split_transform(
                test_data,
                feat_processor,
                label_binarizer,
                data_config=self.data_config,
            )

            for n_repet in range(1, self.evaluation_config.n_repet + 1):
                print(
                    f"Model {model_config.name} | Repet [{n_repet}/{self.evaluation_config.n_repet}]"
                )

                model = initialize_model(model_config=model_config)

                train_model(
                    model_config=model_config,
                    x_train=x_train,
                    y_train=y_train,
                    weights=w_train,
                    model=model,
                )

                train_perf_0, train_perf_1 = evaluate_model(
                    model=model,
                    x=x_train,
                    y=y_train,
                    weights=w_train,
                    data_set="train",
                    model_name=model_config.name,
                )

                test_perf_0, test_perf_1 = evaluate_model(
                    model=model,
                    x=x_test,
                    y=y_test,
                    weights=w_test,
                    data_set="test",
                    model_name=model_config.name,
                )

                data += [
                    {**perf, "repetition": n_repet}
                    for perf in [
                        train_perf_0,
                        train_perf_1,
                        test_perf_0,
                        test_perf_1,
                    ]
                ]

        analyze_data(
            data_dict=data,
            idx_class=1,
            folder_path=self.evaluation_config.folder_out_result,
        )
        analyze_data(
            data_dict=data,
            idx_class=0,
            folder_path=self.evaluation_config.folder_out_result,
        )
