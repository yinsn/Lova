import logging
from typing import Dict

from implicit.als import AlternatingLeastSquares
from numpy import ndarray
from pandas.core.frame import DataFrame

from ..preprocessors import InteractionPreprocessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImplicitALSRecommender(InteractionPreprocessor):
    def __init__(
        self,
        dataset: DataFrame,
        user_column: str,
        item_column: str,
        label_column: str,
        numerical_strength_dict: Dict[str, float],
        bool_strength_vector: ndarray,
        numerical_bool_ratio: float = 0.5,
        percentile: float = 0.999,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 2.0,
        iterations: int = 15,
    ) -> None:
        """
        Initializes the ImplicitALSRecommender.

        Args:
            dataset (DataFrame): The dataset containing user-item interactions.
            user_column (str): The column name for user IDs.
            item_column (str): The column name for item IDs.
            label_column (str): The column name for labels.
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
            user_column,
            item_column,
            label_column,
            numerical_strength_dict,
            bool_strength_vector,
            numerical_bool_ratio,
            percentile,
        )
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
        )
        self.model_prepared = True
        logger.info("Model initialized.")

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
