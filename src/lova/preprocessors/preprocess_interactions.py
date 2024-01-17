import logging
from abc import abstractmethod
from typing import Dict, Optional

import numpy as np
from pandas.core.frame import DataFrame
from scipy.sparse import csr_matrix

from ..aggregators import (
    merge_bool_interactions_with_strength,
    merge_numerical_interactions_with_strength,
    normalize_with_percentile_cap,
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
        config: Optional[Dict] = None,
        user_column: Optional[str] = None,
        item_column: Optional[str] = None,
        label_column: Optional[str] = None,
        numerical_strength_dict: Optional[Dict[str, float]] = {},
        bool_strength_vector: Optional[np.ndarray] = np.array([]),
        numerical_bool_ratio: Optional[float] = 0.5,
        percentile: Optional[float] = 0.999,
    ) -> None:
        """
        Initializes the InteractionPreprocessor.

        Args:
            dataset (DataFrame): The dataset containing user-item interactions.
            config (Dict, optional): A dictionary containing the configuration parameters.
            user_column (str, optional): The column name for user IDs.
            item_column (str, optional): The column name for item IDs.
            label_column (str, optional): The column name for labels.
            numerical_strength_dict (Dict[str, float]): A dictionary mapping numerical interaction types to strengths.
            bool_strength_vector (np.ndarray): A vector indicating strengths for boolean interactions.
            numerical_bool_ratio (float, optional): The ratio for combining numerical and boolean strengths. Defaults to 0.5.
            percentile (float, optional): The percentile to consider for interaction strength calculation. Defaults to 0.999.
        """
        super().__init__(dataset, config, user_column, item_column, label_column)
        if config is not None:
            self.percentile = config.get("percentile", 0.999)
            self.strength_dict = config.get("numerical_strength_dict", {})
            self.strength_vector = config.get("bool_strength_vector", np.array([]))
            self.numerical_bool_ratio = config.get("numerical_bool_ratio", 0.5)
        else:
            self.percentile = percentile
            self.strength_dict = numerical_strength_dict
            self.strength_vector = bool_strength_vector
            self.numerical_bool_ratio = numerical_bool_ratio
        self.preprocess()

    @abstractmethod
    def _merge_sequences(self) -> None:
        """
        Merges sequences of interactions into a single interaction.
        """
        logger.info("Merging sequences...")
        raise NotImplementedError

    def _normalize_with_percentile_cap(
        self,
    ) -> None:
        """
        Normalizes specified columns in the dataset using a percentile cap.

        This method applies normalization to the columns specified in `self.strength_dict`
        within the `self.dataset` DataFrame. The normalization process is capped at the specified percentile,
        meaning values are normalized within the range determined by this percentile.
        """
        self.dataset = normalize_with_percentile_cap(
            dataframe=self.dataset,
            selected_columns=list(self.strength_dict.keys()),
            percentile=self.percentile,
        )

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
        self.rows = self.dataset[self.user_column].map(self.user_id_to_index)
        self.cols = self.dataset[self.item_column].map(self.item_id_to_index)

    def _get_sparse_interaction_matrix(self) -> None:
        """
        Creates a sparse interaction matrix from the dataset.

        This method constructs a CSR (Coordinate) format
        sparse matrix representing user-item interactions.
        """
        logger.info("Creating sparse interaction matrix...")
        data = self.dataset["strength"]
        sparse_matrix = csr_matrix(
            (data, (self.rows, self.cols)),
            shape=(len(self.user_id_to_index), len(self.item_id_to_index)),
        )
        self.sparse_interaction_matrix = sparse_matrix
        logger.info("Creating sparse interaction matrix... Done!")

    def _update_sparse_interaction_matrix(
        self,
        numerical_strength_dict: Dict[str, float],
        bool_strength_vector: np.ndarray,
        numerical_bool_ratio: float = 0.5,
    ) -> None:
        """
        Updates the sparse interaction matrix with new strength parameters.

        This method sets new strength calculation parameters and updates the sparse interaction matrix
        accordingly. It involves recalculating the strength of interactions and reconstructing the sparse matrix
        based on the new strength values.

        Args:
            numerical_strength_dict (Dict[str, float]): A dictionary mapping numerical interaction types to strengths.
            bool_strength_vector (np.ndarray): A vector indicating strengths for boolean interactions.
            numerical_bool_ratio (float, optional): The ratio for combining numerical and boolean strengths. Defaults to 0.5.
        """
        self.strength_dict = numerical_strength_dict
        self.strength_vector = bool_strength_vector
        self.numerical_bool_ratio = numerical_bool_ratio
        self._calculate_strength()
        self._get_sparse_interaction_matrix()

    def preprocess(self) -> None:
        """
        Executes preprocessing steps for the recommender system.

        This method performs the necessary preprocessing steps to prepare
        the data for the recommender system. It includes calculating
        interaction strengths and creating a sparse interaction matrix.
        """
        self._merge_sequences()
        self._normalize_with_percentile_cap()
        self._calculate_strength()
        self._get_id_index_mapping()
        self._get_sparse_interaction_matrix()
