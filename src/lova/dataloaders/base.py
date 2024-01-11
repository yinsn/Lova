from abc import ABCMeta, abstractmethod

import pandas as pd


class BaseDataLoader(metaclass=ABCMeta):
    def __init__(self, file_path: str, file_name: str, file_type: str) -> None:
        self.file_path = file_path
        self.file_name = file_name
        self.file_type = file_type

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError
