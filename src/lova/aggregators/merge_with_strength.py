import logging
from typing import Dict

import numpy as np
import pandas as pd

from .convert_label_encoding_to_vector import label_list_to_vector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_numerical_interactions_with_strength(
    interactions: pd.DataFrame,
    strength_dict: Dict[str, float],
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
    logger.info("Merging interactions with strength values...")
    interactions["numerical_strength"] = np.dot(
        interactions[selected_columns], strength_list
    )
    return interactions


def merge_bool_interactions_with_strength(
    interactions: pd.DataFrame,
    strength_vector: np.ndarray,
    label_column: str,
) -> pd.DataFrame:
    """
    Merges boolean interactions with a strength vector and updates the DataFrame.

    This function takes a DataFrame containing boolean interactions, a specified
    column name containing labels, and a numpy array representing strength vectors.
    It converts the labels in the specified column to vectors using the 'label_list_to_vector'
    function, and then calculates the dot product of these vectors with the strength vector.
    The result is added to the DataFrame as a new column 'bool_strength'.

    Args:
        interactions (pd.DataFrame): The DataFrame containing boolean interactions.
        strength_vector (np.ndarray): A numpy array representing the strength vector.
        label_column (str): The name of the column in the DataFrame which contains the labels.

    Returns:
        pd.DataFrame: The updated DataFrame with a new column 'bool_strength' which is
                      the result of the dot product of label vectors and the strength vector.
    """
    strength_list = []
    logger.info("Merging boolean interactions with strength values...")
    for label in interactions[label_column].to_list():
        strength_list.append(label_list_to_vector(label, len(strength_vector)))
    strength_block = np.stack(strength_list)
    logger.info("Calculating dot product...")
    interactions["bool_strength"] = np.dot(strength_block, strength_vector)
    return interactions
