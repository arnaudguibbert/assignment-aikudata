from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from assaiku.data import DataConfig
from assaiku.model.configs import ModelBaseConfig

from .binarizer import StaticLabelBinarizer


def initialize_feat_processor(
    data_config: DataConfig, model_config: ModelBaseConfig
) -> tuple[Pipeline, StaticLabelBinarizer]:
    # Feature transformations
    transformations = [
        ("standard_norm", StandardScaler(), data_config.numerical_cols),
        (
            "one_hot_encoder",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            data_config.categorical_cols,
        ),
    ]

    ct = ColumnTransformer(
        transformations,
        remainder="drop",
    )

    pipeline_components = [
        ("feature_preprocessor", ct),
    ]

    if model_config.rbf_gamma is not None:
        pipeline_components.append(
            (
                "rbf_sampler",
                RBFSampler(gamma=model_config.rbf_gamma),
            )
        )

    if model_config.dimension_red is not None:
        pipeline_components.append(
            ("dim_reduction", PCA(n_components=model_config.dimension_red))
        )

    pipeline = Pipeline(pipeline_components)
    label_binarizer = StaticLabelBinarizer(classes=data_config.label_values)

    return pipeline, label_binarizer
