#!/usr/bin/env python3
"""
function that converts a one-hot matrix into a vector of labels
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded matrix into a vector of numeric labels.

    Parameters:
    - one_hot (np.ndarray): shape (classes, m), one-hot encoded labels.

    Returns:
    - np.ndarray of shape (m,): numeric labels corresponding to the
    one-hot encoding.
    - None if the input is invalid.
    """
    # Validate input
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    # Use argmax to find the index of the '1' in each column
    labels = np.argmax(one_hot, axis=0)

    return labels
