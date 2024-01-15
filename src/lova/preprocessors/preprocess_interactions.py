import logging
from typing import Dict

import numpy as np
from pandas.core.frame import DataFrame

from ..aggregators import (
    merge_bool_interactions_with_strength,
    merge_numerical_interactions_with_strength,
)
from .base import BasePreprocessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InteractionPreprocessor(BasePreprocessor):
    def __init__(
        self,
        dataset: DataFrame,
        user_column: str,
        item_column: str,
        label_column: str,
        numerical_strength_dict: Dict[str, float],
        bool_strength_vector: np.ndarray,
        numerical_bool_ratio: float = 0.5,
        percentile: float = 0.999,
    ) -> None:
        super().__init__(user_column, item_column, label_column, dataset)
        self.percentile = percentile
        self.strength_dict = numerical_strength_dict
        self.strength_vector = bool_strength_vector
        self.numerical_bool_ratio = numerical_bool_ratio
        self.preprocess()

    def _calculate_strength(self) -> None:
        """
        Calculates and updates the 'strength' of interactions in the dataset.

        This method first merges numerical and boolean interactions with their respective strengths
        using the provided `strength_dict` and `strength_vector`. It then calculates a combined
        'strength' value for each interaction by combining the 'numerical_strength' and
        'bool_strength' based on the `numerical_bool_ratio`. The result is stored in a new
        column named 'strength' in the dataset. The dataset is also filtered to include only
        the user, item, and strength columns. This method logs the process at the start and
        upon completion.

        The method uses the `merge_numerical_interactions_with_strength` and
        `merge_bool_interactions_with_strength` functions from the `aggregators` module for merging
        numerical and boolean interactions, respectively.

        Modifies:
            self.dataset (DataFrame): The dataset is modified to include the 'strength' column and
                                    is filtered to only include the necessary columns.
        """
        self.dataset = merge_numerical_interactions_with_strength(
            interactions=self.dataset,
            strength_dict=self.strength_dict,
            percentile=self.percentile,
        )
        self.dataset = merge_bool_interactions_with_strength(
            interactions=self.dataset,
            strength_vector=self.strength_vector,
            label_column=self.label_column,
        )
        logger.info("Calculating strength...")
        self.dataset["strength"] = (
            self.dataset["numerical_strength"] * self.numerical_bool_ratio
            + self.dataset["bool_strength"]
        )
        self.dataset = self.dataset[[self.user_column, self.item_column, "strength"]]
        logger.info("Calculating strength... Done!")

    def _get_sparse_interaction_matrix(self) -> None:
        pass

    def preprocess(self) -> None:
        self._calculate_strength()
        self._get_sparse_interaction_matrix()
