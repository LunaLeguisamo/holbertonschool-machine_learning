#!/usr/bin/env python3
"""
Function that calculates the total intra-cluster variance for a data set
"""
import numpy as np


def variance(X, C):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster
    You are not allowed to use any loops
    Returns: var, or None on failure
    var is the total variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    dis_min = np.min(distances, axis=1)
    var = np.sum(dis_min ** 2)
    return var
