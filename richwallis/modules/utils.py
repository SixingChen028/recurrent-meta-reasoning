import numpy as np

def merge(lst):
    """
    Merge identical adjacent elements in a list.
    Returns merged values, their lengths, and the start indices.
    """

    if len(lst) == 0:
        return [], [], []

    merged_list = [lst[0]]
    length_list = [1]
    start_indices = [0]

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            length_list[-1] += 1
        else:
            merged_list.append(lst[i])
            length_list.append(1)
            start_indices.append(i)

    return merged_list, length_list, start_indices


def count_states_and_lengths(posterior, min_length = 1, threshold = 0.5):

    all_states, all_lengths, _ = merge(posterior >= threshold)
    all_states = np.array(all_states)
    all_lengths = np.array(all_lengths)

    indices = np.where((all_states == True) & (all_lengths >= min_length))[0]

    states = all_states[indices]
    lengths = all_lengths[indices]

    return len(states), lengths


def count_transitions(posteriors, min_length = 3, threshold = 0.5):

    is_state = posteriors > threshold

    states = []
    i = 0
    while i < len(is_state):
        row = is_state[i]

        # Count consecutive rows of the same type
        j = i
        while j < len(is_state) and np.array_equal(is_state[j], row):
            j += 1
        run_length = j - i

        if np.array_equal(row, [True, False]) and run_length >= min_length:
            states.append(0)
        elif np.array_equal(row, [False, True]) and run_length >= min_length:
            states.append(1)
        # [False, False] stretches are ignored

        i = j  # move to next different block

    transitions = merge(states)[0]

    return len(transitions) - 1


def find_start_indices_and_lengths(posterior, min_length = 1, threshold = 0.5):

    all_states, all_lengths, all_starts = merge(posterior >= threshold)
    all_states = np.array(all_states)
    all_lengths = np.array(all_lengths)
    all_starts = np.array(all_starts)

    indices = np.where((all_states == True) & (all_lengths >= min_length))[0]

    starts = all_starts[indices]
    lengths = all_lengths[indices]

    return starts, lengths