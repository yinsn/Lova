import logging

import pandas as pd

from ..recommenders import ImplicitALSRecommender
from .base import BaseEvaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImplicitALSEvaluator(BaseEvaluator):
    def __init__(
        self,
        dataset: pd.DataFrame,
        recommender: ImplicitALSRecommender,
    ) -> None:
        """
        Initializes the ImplicitALSEvaluator.

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation.
            recommender (ImplicitALSRecommender): The recommender system instance to evaluate.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.recommender = recommender
        self._map_id_to_index()

    def _map_id_to_index(self) -> None:
        """
        Maps user and item IDs to indices based on the recommender's mappings.

        This method creates two new columns in the dataset, 'user_map' and 'item_map', which contain
        the mapped indices of users and items, respectively. It also drops any NaN values after mapping.
        """
        logger.info("Mapping IDs to indices...")
        self.dataset["user_map"] = self.dataset[self.recommender.user_column].map(
            self.recommender.user_id_to_index
        )
        self.dataset["item_map"] = self.dataset[self.recommender.item_column].map(
            self.recommender.item_id_to_index
        )
        logger.info("Dropping NaN values...")
        self.dataset.dropna(inplace=True)
        self.user_evaluate_indice = self.dataset.user_map.astype(int).to_list()
        self.item_evaluate_indice = self.dataset.item_map.astype(int).to_list()

    def evaluate(self) -> float:
        """
        Evaluates the recommender system.

        Calculates a score by computing the dot product of user and item vectors obtained from the recommender.
        The mean of these scores is returned as the final evaluation metric.
        """
        logger.info("Evaluating...")
        user_vectors = self.recommender.model.user_factors[self.user_evaluate_indice]
        item_vectors = self.recommender.model.item_factors[self.item_evaluate_indice]
        score = (user_vectors * item_vectors).sum(axis=1)
        return float(score.mean())
