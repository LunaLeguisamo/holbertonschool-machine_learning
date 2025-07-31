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
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n = X.shape[0]

    # Si no se pasa kmax, lo establezco en n
    if kmax is None:
        kmax = n

    # Validaciones de parámetros
    if not isinstance(kmin, int) or kmin < 1:
        return None, None

    if not isinstance(kmax, int) or kmax < 1 or kmax > n:
        return None, None

    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    if kmax < kmin:
        return None, None

    results = []
    d_vars = []

    # Cálculo de la varianza base con kmin clusters
    C_min, _ = kmeans(X, kmin, iterations)
    if C_min is None:
        return None, None
    var_min = variance(X, C_min)
    if var_min is None:
        return None, None

    # Loop principal desde kmin hasta kmax
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        var = variance(X, C)
        if var is None:
            return None, None

        results.append((C, clss))
        d_vars.append(var_min - var)

    return results, d_vars
