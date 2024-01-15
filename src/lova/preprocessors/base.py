from abc import ABCMeta, abstractmethod

import pandas as pd


class BasePreprocessor(metaclass=ABCMeta):
    def __init__(
        self,
        user_column: str,
        item_column: str,
        label_column: str,
        dataset: pd.DataFrame,
    ) -> None:
        self.user_column = user_column
        self.item_column = item_column
        self.label_column = label_column
        self.dataset = dataset

    @abstractmethod
    def preprocess(self) -> None:
        raise NotImplementedError
