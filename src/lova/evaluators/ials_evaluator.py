import logging
from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial

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
        study_name: Optional[str] = None,
        study_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the ImplicitALSEvaluator.

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation.
            study_name (str, optional): The name of the study directory. If not provided, the current
                                        system time in the format 'YYYY_MM_DD_HH_MM' will be used.
            study_path (str, optional): The base directory path. Defaults to the current directory.
            recommender (ImplicitALSRecommender): The recommender system instance to evaluate.
        """
        super().__init__(dataset, study_name, study_path)
        self.recommender = recommender
        self._map_id_to_index()
        self.study = optuna.create_study(direction="maximize")

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

    def _update_recommender(
        self,
        numerical_strength_dict: Dict[str, float],
        bool_strength_vector: np.ndarray,
        numerical_bool_ratio: float,
    ) -> float:
        """
        Updates the recommender's internal state based on the provided data.

        This method updates the sparse interaction matrix of the recommender system using
        a combination of numerical strengths and boolean strengths, controlled by a ratio
        parameter. After updating the interaction matrix, the recommender is refitted.

        Args:
            numerical_strength_dict (Dict[str, float]): A dictionary mapping item identifiers
                to their numerical strengths. These strengths represent the magnitude of
                interaction or preference a user has towards these items.
            bool_strength_vector (np.ndarray): A numpy array representing boolean strengths.
                This vector indicates the presence or absence of interaction or preference
                between a user and items.
            numerical_bool_ratio (float): A ratio determining the balance between numerical
                strengths and boolean strengths in updating the interaction matrix.
        """
        self.recommender._update_sparse_interaction_matrix(
            numerical_strength_dict=numerical_strength_dict,
            bool_strength_vector=bool_strength_vector,
            numerical_bool_ratio=numerical_bool_ratio,
        )
        self.recommender.fit()
        return self.evaluate()

    def evaluate(self) -> float:
        """
        Evaluates the recommender system.

        Calculates a score by computing the dot product of user and item vectors obtained from the recommender.
        The mean of these scores is returned as the final evaluation metric.
        """
        logger.info("Evaluating...")
        user_vectors = self.recommender.model.user_factors[self.user_evaluate_indice]
        item_vectors = self.recommender.model.item_factors[self.item_evaluate_indice]
        scores = (user_vectors * item_vectors).sum(axis=1)
        score = float(scores.mean())
        logger.info(f"Evaluation score: {score}")
        return score

    def objective(self, trial: Trial) -> float:
        """
        Optimizes the recommender system's weights based on the trial configurations provided by Optuna.

        This function is designed to be used as an objective function for an Optuna optimization study. It suggests
        values for `numerical_bool_ratio`, `numerical_weights`, and `bool_weights`, and then uses these values to
        update the recommender system. The performance of the recommender system with these weights is evaluated,
        and the resulting score is returned as the objective value.

        Args:
            trial (Trial): An Optuna trial object which provides methods to suggest parameters, such as floats, ints, and categoricals.

        Returns:
            float: The evaluation score of the recommender system with the suggested parameters.
        """
        numerical_bool_ratio = trial.suggest_float(
            "numerical_bool_ratio", 1e-1, 1e3, log=True
        )
        numerical_weights = [
            trial.suggest_float(f"{key}", 1e-1, 1e3, log=True)
            for key in self.recommender.strength_dict.keys()
        ]
        bool_weights = [
            trial.suggest_float(f"bool_weights{i}", 1e-1, 1e3, log=True)
            for i in range(len(self.recommender.strength_vector))
        ]
        strength_dict = {}
        for index, key in enumerate(self.recommender.strength_dict.keys()):
            strength_dict[key] = numerical_weights[index]
        strength_vector = np.array(bool_weights)
        logger.info(f"Trial number: {trial.number}")
        logger.info(f"numerical_bool_ratio: {numerical_bool_ratio}")
        logger.info(f"numerical_weights: {numerical_weights}")
        logger.info(f"bool_weights: {bool_weights}")
        score = self._update_recommender(
            numerical_strength_dict=strength_dict,
            bool_strength_vector=strength_vector,
            numerical_bool_ratio=numerical_bool_ratio,
        )
        return score

    def tune(self, n_trials: int = 100) -> None:
        """
        Conducts a hyperparameter tuning session using the Optuna framework.

        This method initializes a logger, then uses Optuna's `study.optimize` method to optimize the recommender system's
        parameters. The optimization is guided by the `objective` method of this class, which is expected to be defined elsewhere
        within the class.

        Args:
            n_trials (int, optional): The number of trials for optimization. Defaults to 100.
        """
        self.build_logger()
        self.study.optimize(self.objective, n_trials=n_trials)
