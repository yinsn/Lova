from abc import ABCMeta, abstractmethod

import pandas as pd


class BaseEvaluator(metaclass=ABCMeta):
    """
    Abstract base class for creating evaluator classes.

    This class serves as a template for creating evaluator classes that assess the performance
    of various algorithms or models using a given dataset.

    Attributes:
        dataset (pd.DataFrame): The dataset used for evaluation.
    """

    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        Initializes the BaseEvaluator with a dataset.

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation.
        """
        self.dataset = dataset

    @abstractmethod
    def evaluate(self) -> float:
        """
        Abstract method to be implemented for evaluating an algorithm or model.

        This method should be overridden in subclasses to provide specific evaluation logic.

        Returns:
            float: The evaluation result.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
