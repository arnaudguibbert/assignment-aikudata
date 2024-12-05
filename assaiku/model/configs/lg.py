from pydantic import Field

from .base import ModelBaseConfig


class LogisticRegressionConfig(ModelBaseConfig):
    max_iter: int = Field(200, gt=1)

    @property
    def name(self) -> str:
        return f"LogReg(dimred={self.dimension_red is not None})"
