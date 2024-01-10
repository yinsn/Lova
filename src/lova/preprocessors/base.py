from abc import ABCMeta, abstractmethod

import pandas as pd


class BasePreprocessor(metaclass=ABCMeta):
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset

    @abstractmethod
    def preprocess(self) -> None:
        raise NotImplementedError
