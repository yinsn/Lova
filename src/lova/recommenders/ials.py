import logging
import pickle
from typing import Dict, Optional

import numpy as np
from implicit.als import AlternatingLeastSquares
from pandas.core.frame import DataFrame

from ..aggregators import sequence_with_equal_importance
from ..dataloaders import ensure_study_directory
from ..preprocessors import InteractionPreprocessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImplicitALSRecommender(InteractionPreprocessor):
    def __init__(
        self,
        dataset: DataFrame,
        config: Dict,
        user_column: Optional[str] = None,
        item_column: Optional[str] = None,
        label_column: Optional[str] = None,
        numerical_strength_dict: Optional[Dict[str, float]] = {},
        bool_strength_vector: Optional[np.ndarray] = np.array([]),
        numerical_bool_ratio: Optional[float] = 0.5,
        percentile: Optional[float] = 0.999,
        factors: Optional[int] = 64,
        regularization: Optional[float] = 0.05,
        alpha: Optional[float] = 2.0,
        iterations: Optional[int] = 15,
    ) -> None:
        """
        Initializes the ImplicitALSRecommender.

        Args:
            dataset (DataFrame): The dataset containing user-item interactions.
            config (Dict, optional): A dictionary containing the configuration parameters.
            user_column (str, optional): The column name for user IDs.
            item_column (str, optional): The column name for item IDs.
            label_column (str, optional): The column name for labels.
            numerical_strength_dict (Dict[str, float]): A dictionary mapping numerical interaction types to strengths.
            bool_strength_vector (ndarray): A vector indicating strengths for boolean interactions.
            numerical_bool_ratio (float, optional): The ratio for combining numerical and boolean strengths. Defaults to 0.5.
            percentile (float, optional): The percentile to consider for interaction strength calculation. Defaults to 0.999.
            factors (int, optional): The number of latent factors to compute. Defaults to 64.
            regularization (float, optional): The regularization parameter for ALS. Defaults to 0.05.
            alpha (float, optional): The confidence weight, higher values indicate more confidence. Defaults to 2.0.
            iterations (int, optional): The number of ALS iterations to run. Defaults to 15.
        """
        super().__init__(
            dataset,
            config,
            user_column,
            item_column,
            label_column,
            numerical_strength_dict,
            bool_strength_vector,
            numerical_bool_ratio,
            percentile,
        )
        if config is not None:
            self.factors = config.get("factors", 64)
            self.regularization = config.get("regularization", 0.05)
            self.alpha = config.get("alpha", 2.0)
            self.iterations = config.get("iterations", 15)
        else:
            self.factors = factors
            self.regularization = regularization
            self.alpha = alpha
            self.iterations = iterations
        self.model_prepared = False

    def _init_model(self) -> None:
        """
        Initializes the ALS model with the specified parameters.

        This method creates an instance of the AlternatingLeastSquares model from the implicit library
        using the parameters provided during the object initialization. It sets the `model_prepared` flag to True after
        initialization.
        """
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=42,
        )
        self.model_prepared = True
        logger.info("Model initialized.")

    def _merge_sequences(self) -> None:
        """
        Merges sequences in the dataset with equal importance.

        This method applies the `sequence_with_equal_importance` function to the dataset, focusing on
        the columns specified in `self.strength_dict`. The purpose of this merging is to combine
        sequences based on certain criteria defined in `sequence_with_equal_importance`.
        """
        sequence_with_equal_importance(
            interactions=self.dataset,
            selected_columns=list(self.strength_dict.keys()),
        )

    def fit(self) -> None:
        """
        Fits the ALS model to the interaction data.

        This method checks if the model is prepared; if not, it initializes the model. Then, it fits the ALS model
        to the sparse interaction matrix created during the preprocessing phase. The item and user factors are
        extracted after fitting the model.
        """
        if self.model_prepared == False:
            self._init_model()
        logger.info("Fitting model...")
        self.model.fit(user_items=self.sparse_interaction_matrix)
        logger.info("Model fitted.")
        self.item_factors = self.model.item_factors
        self.user_factors = self.model.user_factors

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Dumps the model to a file.

        This method dumps the model to a file using the pickle library.

        Args:
            path (str): The path to the file to dump the model to.
        """
        if path is None and self.config is not None:
            path = ensure_study_directory(
                study_name=self.config.get("study_name", None),
                study_path=self.config.get("study_path", None),
            )
        logger.info(f"Saving model in {path}...")
        logger.info("Saving user id to index mapping...")
        with open(f"{path}/user_id_to_index.pkl", "wb") as f:
            pickle.dump(self.user_id_to_index, f)
        logger.info("Saving item id to index mapping...")
        with open(f"{path}/item_id_to_index.pkl", "wb") as f:
            pickle.dump(self.item_id_to_index, f)
        logger.info("Saving user factors...")
        with open(f"{path}/user_factors.pkl", "wb") as f:
            pickle.dump(self.user_factors, f)
        logger.info("Saving item factors...")
        with open(f"{path}/item_factors.pkl", "wb") as f:
            pickle.dump(self.item_factors, f)
