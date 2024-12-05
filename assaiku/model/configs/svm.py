from pydantic import Field

from .base import ModelBaseConfig


class LinearSVMConfig(ModelBaseConfig):
    max_iter: int = Field(1500, gt=1)
    C: float = Field(1, gt=0.0)

    @property
    def name(self) -> str:
        return f"LinearSVC(rbf={self.rbf_gamma is not None})"
