#!/usr/bin/env python3
"""
function that converts a numeric label vector
into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Parameters:
    - Y (np.ndarray): shape (m,), contains numeric class labels.
    - classes (int): total number of possible classes.

    Returns:
    - np.ndarray of shape (classes, m): one-hot encoded matrix.
    - None if input is invalid.
    """
    # Validate input
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    m = Y.shape[0]  # number of examples

    # Initialize the output matrix with zeros
    one_hot = np.zeros((classes, m))

    # Set the correct positions to 1 using advanced indexing
    one_hot[Y, np.arange(m)] = 1
    return one_hot
