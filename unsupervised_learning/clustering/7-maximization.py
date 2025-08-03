#!/usr/bin/env python3
"""
Function that calculates the maximization
step in the EM algorithm for a GMM
"""

import numpy as np


def maximization(X, g):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the updated priors for
    each cluster
    m is a numpy.ndarray of shape (k, d) containing the updated centroid
    means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the updated covariance
    matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    N_k = np.sum(g, axis=1)
    pi = N_k / n
    m = (g @ X) / N_k[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted_diff = diff.T * g[i]
        S[i] = weighted_diff @ diff / N_k[i]

    return pi, m, S
