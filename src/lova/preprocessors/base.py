from abc import ABCMeta, abstractmethod

import pandas as pd


class BasePreprocessor(metaclass=ABCMeta):
    """
    An abstract base class for preprocessing datasets.

    This class is intended to be subclassed with specific implementations of
    the `preprocess` method for different preprocessing strategies.

    Attributes:
        user_column (str): The name of the user column in the dataset.
        item_column (str): The name of the item column in the dataset.
        label_column (str): The name of the label column in the dataset.
        dataset (pd.DataFrame): The dataset to be preprocessed.
    """

    def __init__(
        self,
        user_column: str,
        item_column: str,
        label_column: str,
        dataset: pd.DataFrame,
    ) -> None:
        """
        Initializes the BasePreprocessor with the dataset and column names.

        Args:
            user_column (str): The name of the user column in the dataset.
            item_column (str): The name of the item column in the dataset.
            label_column (str): The name of the label column in the dataset.
            dataset (pd.DataFrame): The dataset to be preprocessed.
        """
        self.user_column = user_column
        self.item_column = item_column
        self.label_column = label_column
        self.dataset = dataset

    @abstractmethod
    def preprocess(self) -> None:
        """
        Abstract method for preprocessing the dataset.

        This method should be implemented by subclasses to apply specific preprocessing steps.
        The implementation should modify the `self.dataset` attribute in-place.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError
