#!/usr/bin/env python3
"""
Function that shuffles the data points in two matrices
(X and Y)
in the same way to preserve the input-label correspondence.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles X and Y in unison, preserving the correspondence
    between
    each sample in X and its label in Y.

    Parameters:
    -----------
    X : numpy.ndarray of shape (m, nx)
        The data matrix to shuffle. m is the number of
        examples,
        nx is the number of features per example.

    Y : numpy.ndarray of shape (m, ny)
        The labels matrix to shuffle. ny is the number of
        label components.

    Returns:
    --------
    X_shuffled, Y_shuffled : tuple of numpy.ndarrays
        The shuffled data and labels.
    """
    # Generate a permutation of the indices from 0 to m-1
    permutation = np.random.permutation(X.shape[0])

    # Apply the same permutation to both X and Y
    return X[permutation], Y[permutation]
