import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

import optuna
import pandas as pd

from ..dataloaders.set_path import ensure_study_directory


class BaseEvaluator(metaclass=ABCMeta):
    """
    Abstract base class for creating evaluator classes.

    This class serves as a template for creating evaluator classes that assess the performance
    of various algorithms or models using a given dataset.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        study_name: Optional[str] = None,
        study_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the BaseEvaluator with a dataset.

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation.
            study_name (str, optional): The name of the study directory. If not provided, the current
                                        system time in the format 'YYYY_MM_DD_HH_MM' will be used.
            study_path (str, optional): The base directory path. Defaults to the current directory.
        """
        self.dataset = dataset
        self.study_name = study_name
        self.study_path = study_path
        self.full_path = ensure_study_directory(study_path, study_name)

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

    def build_logger(self) -> None:
        """
        Constructs a logger for the optimization process.
        """
        log_filename = f"{self.full_path}/lova.log"

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        self.logger = optuna.logging.get_logger(f"lova")
        self.logger.addHandler(file_handler)
