from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from ..configs import (
    LinearSVMConfig,
    LogisticRegressionConfig,
    ModelBaseConfig,
    XGBConfig,
)


def initialize_model(model_config: ModelBaseConfig):
    match model_config:
        case LogisticRegressionConfig():
            clf = LogisticRegression(
                max_iter=model_config.max_iter,
            )
        case LinearSVMConfig():
            clf = LinearSVC(
                max_iter=model_config.max_iter,
                C=model_config.C,
            )
        case XGBConfig():
            clf = XGBClassifier(
                n_estimators=model_config.n_estimators,
                max_depth=model_config.max_depth,
                learning_rate=model_config.learning_rate,
                objective="binary:logistic",
            )
        case _:
            raise ValueError(f"Unexpected config {model_config}")

    return clf
