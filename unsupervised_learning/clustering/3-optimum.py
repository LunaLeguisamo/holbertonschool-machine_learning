#!/usr/bin/env python3
"""
Function that tests for the optimum number of clusters by variance
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters to
    check for
    kmax is a positive integer containing the maximum number of clusters to
    check for
    iterations is a positive integer containing the maximum number of
    iterations for K-means
    This function should analyze at least 2 different cluster sizes
    Returns: results, d_vars, or None, None on failure
    results is a list containing the outputs of K-means for each cluster size
    d_vars is a list containing the difference in variance from the smallest
    cluster size for each cluster size
    """
    if not isinstance(kmin, int) or kmin < 0:
        return None, None

    if not isinstance(kmax, int) or kmax < 0:
        return None, None

    if not isinstance(iterations, int) or iterations < 0:
        return None, None

    if not kmax >= kmin:
        return None, None

    if kmin < 1:
        return None, None

    results = []
    d_vars = []

    C_min, _ = kmeans(X, kmin, iterations)
    var_min = variance(X, C_min)

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        results.append((C, clss))
        d_vars.append(var_min - var)

    return results, d_vars
