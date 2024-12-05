from pydantic import BaseModel

from assaiku.model.configs import ModelBaseConfig

from .lg import LogisticRegressionConfig
from .svm import LinearSVMConfig
from .xgboost import XGBConfig


class EvaluationConfig(BaseModel):
    n_repet: int = 1
    model_configs: list[ModelBaseConfig] = [
        XGBConfig(),
        LinearSVMConfig(rbf_feataug=True, balance_weights=True),
        # LogisticRegressionConfig(),
        # LogisticRegressionConfig(dimension_red=20),
    ]
