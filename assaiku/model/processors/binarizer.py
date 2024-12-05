import numpy as np
from scipy.sparse import spmatrix
from sklearn.base import TransformerMixin
from sklearn.preprocessing import label_binarize


class StaticLabelBinarizer(TransformerMixin):
    def __init__(self, classes: tuple[str, str]):
        super().__init__()
        self._classes = classes

    def fit(self, y) -> "StaticLabelBinarizer":
        return self

    @property
    def classes_(self) -> tuple[str, str]:
        return self._classes

    def transform(self, y) -> np.ndarray | spmatrix:
        return label_binarize(y=y, classes=self._classes).squeeze(1)
