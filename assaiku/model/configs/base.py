from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ModelBaseConfig(BaseModel, ABC):
    dimension_red: int | None = Field(None, gt=0)
    balance_weights: bool = True
    rbf_gamma: float | None = Field(None, gt=0.0)

    weight_pos_factor: float = Field(1.0, gt=0.0)
    weight_neg_factor: float = Field(1.0, gt=0.0)

    @property
    @abstractmethod
    def name(self) -> str:
        pass
