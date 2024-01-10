from pandas.core.frame import DataFrame

from .base import BasePreprocessor


class InteractionPreprocessor(BasePreprocessor):
    def __init__(self, dataset: DataFrame) -> None:
        super().__init__(dataset)

    def _get_sparse_interaction_matrix(self) -> None:
        pass

    def preprocess(self) -> None:
        self._get_sparse_interaction_matrix()
