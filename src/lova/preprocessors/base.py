from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BasePreprocessor(metaclass=ABCMeta):
    """
    An abstract base class for preprocessing datasets.

    This class is intended to be subclassed with specific implementations of
    the `preprocess` method for different preprocessing strategies.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        config: Optional[Dict] = None,
        user_column: Optional[str] = None,
        item_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Initializes the BasePreprocessor with the dataset and column names.

        Args:
            user_column (str, optional): The name of the user column in the dataset.
            item_column (str, optional): The name of the item column in the dataset.
            label_column (str, optional): The name of the label column in the dataset.
            config (Dict, optional): A dictionary containing the configuration parameters.
            dataset (pd.DataFrame): The dataset to be preprocessed.
        """
        self.config = config
        if config is not None:
            self.user_column = config.get("user_column", "")
            self.item_column = config.get("item_column", "")
            self.label_column = config.get("label_column", "")
        else:
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
