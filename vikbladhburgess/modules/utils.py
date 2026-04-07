import numpy as np


def merge(lst):
    """
    Merge identical adjacent elements in a list.
    """

    # check if the list is empty
    if not lst:
        return []

    # start with the first element
    merged_list = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            merged_list.append(lst[i])

    return merged_list
