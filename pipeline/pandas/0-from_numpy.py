#!/usr/bin/env python3
"""
Convert a NumPy array to a Pandas DataFrame.
"""
import pandas as pd
import string


def from_numpy(array):
    """
    that creates a pd.DataFrame from a np.ndarray
    array is the np.ndarray from which you should
    create the pd.DataFrame
    The columns of the pd.DataFrame should be
    labeled in alphabetical order and capitalized.
    There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    n_columns = array.shape[1]
    alpha = list(string.ascii_uppercase[:n_columns])
    data_frame = pd.DataFrame(array, columns=alpha)
    return data_frame
