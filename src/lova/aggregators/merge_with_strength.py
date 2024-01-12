import logging
from typing import Dict

import numpy as np
import pandas as pd

from .normalize_with_percentile import normalize_with_percentile_cap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_numerical_interactions_with_strength(
    interactions: pd.DataFrame,
    strength_dict: Dict[str, float],
    percentile: float = 0.999,
) -> pd.DataFrame:
    """
    Merge interaction data with strength values to compute a combined 'strength' column.

    This function takes a DataFrame of interactions and a dictionary mapping column names to
    strengths. It computes the sum of each specified column in `interactions` and then creates a
    new 'strength' column in the DataFrame. This 'strength' column is the dot product of the
    interaction sums and the strength values.

    Args:
        interactions (pd.DataFrame): A DataFrame containing interaction data.
        strength_dict (Dict[str, float]): A dictionary mapping column names in the DataFrame
                                          to strength values.

    Returns:
        pd.DataFrame: The modified DataFrame with an additional 'strength' column.

    Examples:
        >>> interactions = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> strength_dict = {'A': 0.5, 'B': 1.5}
        >>> merge_interactions_with_strength(interactions, strength_dict)
    """
    selected_columns = []
    strength_list = []
    for key, value in strength_dict.items():
        selected_columns.append(key)
        strength_list.append(value)
    for col in selected_columns:
        interactions[col] = interactions[col].apply(sum)
    interactions = normalize_with_percentile_cap(
        dataframe=interactions,
        selected_columns=selected_columns,
        percentile=percentile,
    )
    logger.info("Merging interactions with strength values...")
    interactions["numerical_strength"] = np.dot(
        interactions[selected_columns], strength_list
    )
    return interactions
