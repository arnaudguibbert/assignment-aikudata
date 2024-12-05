from pydantic import Field

from .base import ModelBaseConfig


class XGBConfig(ModelBaseConfig):
    n_estimators: int = Field(100, ge=1)
    max_depth: int = Field(8, ge=1)
    learning_rate: float = Field(1e-1, gt=0.0)

    @property
    def name(self) -> str:
        return f"XGB(dimred={self.dimension_red is not None})"
