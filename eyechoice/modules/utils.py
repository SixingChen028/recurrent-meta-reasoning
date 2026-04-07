import numpy as np

def merge(lst):
    """
    Merge identical adjacent elements in a list.
    """

    # check if the list is empty
    if not lst:
        return [], []

    # start with the first element
    merged_list = [lst[0]]
    count_list = [1]

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            count_list[-1] += 1  # increment count for the current sequence
        elif lst[i] != lst[i - 1]:
            merged_list.append(lst[i])
            count_list.append(1)  # reset count for the new element

    return merged_list, count_list


def compute_aligned_proportions(array, unique_values):
    """
    Computes proportions of array aligned to unique values.
    Missing values are filled with np.nan.
    
    Args:
        array: an np.array or a list
        unique_values: an np.array or a list.

    Returns:
        aligned_proportions: an np.array. [roportions aligned to unique_values, with np.nan for missing entries.
    """
    # compute unique values and their counts in the data
    uniques, counts = np.unique(array, return_counts = True)
    proportions = counts / np.sum(counts)

    # create a mapping from unique value to proportion
    prop_dict = dict(zip(uniques, proportions))

    # align to uniques_to_plot, inserting np.nan for missing values
    aligned_proportions = np.array([
        prop_dict.get(u, np.nan) for u in unique_values
    ])

    return aligned_proportions


def pad_with_nan(arr, max_len):
    """
    Pad a 1D NumPy array with np.nan up to max_len.
    """

    arr = np.asarray(arr, dtype = float)  # ensure float so np.nan is valid
    if len(arr) < max_len:
        pad_len = max_len - len(arr)
        arr = np.concatenate([arr, np.full(pad_len, np.nan)])
    return arr