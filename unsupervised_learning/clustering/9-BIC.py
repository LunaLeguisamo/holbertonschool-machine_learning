#!/usr/bin/env python3
"""
Function that finds the best number of clusters
for a GMM using the Bayesian Information Criterion
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin < 1 or kmin >= n:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax < kmin or kmax >= n:
        return None, None, None, None

    best_k = None
    best_result = None
    log_likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        results = expectation_maximization(X, k, iterations, tol, verbose)

        if results is None:
            return None, None, None, None

        pi, m, S, g, log_likelihood = results

        # Guardamos el log_likelihood
        log_likelihoods.append(log_likelihood)

        # Cantidad de parámetros
        p = (k * d * (d + 1)) / 2 + d * k + (k - 1)

        # Cálculo del BIC
        bic = p * np.log(n) - 2 * log_likelihood
        bics.append(bic)

        # Guardamos el mejor resultado
        if best_result is None or bic < bics[best_k - kmin]:
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
