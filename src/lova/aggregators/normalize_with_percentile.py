import logging
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_with_percentile_cap(
    dataframe: pd.DataFrame, selected_columns: List[str], percentile: float = 0.999
) -> pd.DataFrame:
    """
    Normalize selected columns in a DataFrame with a percentile cap.

    This function applies a percentile cap to each of the specified columns in the DataFrame.
    Values above the percentile cap are set to the cap value. Then, it normalizes these columns
    within the range 0 to 1 based on the minimum and maximum values after applying the cap.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be normalized.
        selected_columns (List[str]): A list of column names in the DataFrame to be normalized.
        percentile (float, optional): The percentile to use as a cap for normalization. Defaults to 0.999.

    Returns:
        pd.DataFrame: The DataFrame with the selected columns normalized.
    """
    logger.info("Normalizing with percentile cap...")
    for col in selected_columns:
        percentile_value = dataframe[col].quantile(percentile)
        dataframe[col] = dataframe[col].apply(lambda x: min(x, percentile_value))
        dataframe[col] = (dataframe[col] - dataframe[col].min()) / (
            dataframe[col].max() - dataframe[col].min()
        )
    return dataframe
