from typing import List

import numpy as np


def sequence_order_from_date(date_list: List[int]) -> List:
    """
    Determine the order of indices that would sort the date list.

    This function takes a list of dates represented as integers and returns a list of indices.
    These indices represent the order in which the original dates should be arranged to be sorted.

    Args:
        date_list (List[int]): A list of dates represented as integers.

    Returns:
        List[int]: A list of indices representing the sorted order of the dates.

    Examples:
        >>> sequence_order_from_date([20200101, 20210101, 20200102])
        [1, 0, 2]
    """
    sorted_order = np.argsort(date_list)
    positions = np.argsort(sorted_order[::-1])
    return list(positions)
