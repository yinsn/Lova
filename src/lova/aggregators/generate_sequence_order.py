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
        [2, 0, 1]
    """
    sorted_order = np.argsort(date_list)
    positions = np.argsort(sorted_order[::-1])
    return list(positions)


def sequence_order_from_date_with_time_dacay(
    date_list: List[int], time_decay: float = 1
) -> List:
    """
    Sort a list of dates and apply a time decay factor to each sorted position.

    This function first sorts the given list of dates and then applies a time decay factor to each
    position in the sorted list. The time decay factor is raised to the power of the position.

    Args:
        date_list (List[int]): A list of dates represented as integers.
        time_decay (float, optional): The time decay factor to apply to each position. Defaults to 1.

    Returns:
        List[float]: A list of decay-adjusted values based on the sorted order of the dates.

    Examples:
        >>> sequence_order_from_date_with_time_decay([20200101, 20210101, 20200102], 0.5)
        [0.25, 1.0, 0.5]  # Assuming the sorted positions are [2, 0, 1]
    """
    positions = sequence_order_from_date(date_list)
    return [time_decay**order for order in positions]
