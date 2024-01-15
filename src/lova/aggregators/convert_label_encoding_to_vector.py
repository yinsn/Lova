from typing import List

import numpy as np


def label_to_vector(label: int, field_num: int) -> np.ndarray:
    """
    Convert single-element integer list to binary vector using bit-shifting, bounded by 'field_num'

    This function takes a label, represented as a list of integers, and converts it
    into a binary vector. The conversion is done by bit-shifting each integer in the label
    and applying a bitwise AND operation. The resulting binary vector has a length
    specified by 'field_num'.

    Args:
        label (List[int]): A list of integers representing the label.
        field_num (int): The length of the binary vector representation.

    Returns:
        np.ndarray: A numpy array representing the label in binary vector form.
    """
    return np.array([(label >> i) & 1 for i in range(field_num - 1, -1, -1)])


def label_list_to_vector(label_list: List[int], field_num: int) -> np.ndarray:
    """
    Converts a list of labels to a vector representation.

    This function takes a list of labels, each represented as a list of integers, and
    converts them into a vector representation. It does so by calling 'label_to_vector'
    for each label in the list and summing the resulting vectors. The length of the
    vector representation is specified by 'field_num'.

    Args:
        label_list (List[int]): A list of labels, each label is a list of integers.
        field_num (int): The length of the binary vector representation for each label.

    Returns:
        np.ndarray: A numpy array representing the summed vector of all labels.
    """
    vec = np.zeros(field_num)
    for label in label_list:
        vec += label_to_vector(label, field_num)
    return vec
