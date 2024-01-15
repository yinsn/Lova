import logging
from typing import Dict

import numpy as np
from pandas.core.frame import DataFrame
from scipy.sparse import csr_matrix

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
        """
        Initializes the InteractionPreprocessor.

        Args:
            dataset (DataFrame): The dataset containing user-item interactions.
            user_column (str): The column name for user IDs.
            item_column (str): The column name for item IDs.
            label_column (str): The column name for labels.
            numerical_strength_dict (Dict[str, float]): A dictionary mapping numerical interaction types to strengths.
            bool_strength_vector (np.ndarray): A vector indicating strengths for boolean interactions.
            numerical_bool_ratio (float, optional): The ratio for combining numerical and boolean strengths. Defaults to 0.5.
            percentile (float, optional): The percentile to consider for interaction strength calculation. Defaults to 0.999.
        """
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

    def _get_id_index_mapping(self) -> None:
        """
        Creates a mapping from user and item IDs to their respective indices.

        This method updates the `user_id_to_index` and `item_id_to_index` attributes
        of the class with dictionaries mapping IDs to indices.
        """
        logger.info("Creating id to index mapping...")
        user_ids = self.dataset[self.user_column].unique()
        user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        self.user_id_to_index = user_id_to_index
        item_ids = self.dataset[self.item_column].unique()
        item_id_to_index = {item_id: i for i, item_id in enumerate(item_ids)}
        self.item_id_to_index = item_id_to_index

    def _get_sparse_interaction_matrix(self) -> None:
        """
        Creates a sparse interaction matrix from the dataset.

        This method first generates ID to index mappings by calling
        `_get_id_index_mapping` and then constructs a COO (Coordinate) format
        sparse matrix representing user-item interactions.
        """
        logger.info("Creating sparse interaction matrix...")
        self._get_id_index_mapping()
        rows = self.dataset[self.user_column].map(self.user_id_to_index)
        cols = self.dataset[self.item_column].map(self.item_id_to_index)
        data = self.dataset["strength"]
        sparse_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_id_to_index), len(self.item_id_to_index)),
        )
        self.sparse_interaction_matrix = sparse_matrix
        logger.info("Creating sparse interaction matrix... Done!")

    def preprocess(self) -> None:
        """
        Executes preprocessing steps for the recommender system.

        This method performs the necessary preprocessing steps to prepare
        the data for the recommender system. It includes calculating
        interaction strengths and creating a sparse interaction matrix.
        """
        self._calculate_strength()
        self._get_sparse_interaction_matrix()
