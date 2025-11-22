#!/usr/bin/env python3
"""
Function that creates a pd.DataFrame
from a np.ndarray
"""
import pandas as pd
import string

def from_numpy(array):
    """
    array is the np.ndarray from which you
    should create the pd.DataFrame
    The columns of the pd.DataFrame should
    be labeled in alphabetical order and capitalized.
    There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    n_columns = array.shape[1]
    alpha = list(string.ascii_uppercase[:n_columns])
    df = pd.DataFrame(array, columns=alpha)
    return df