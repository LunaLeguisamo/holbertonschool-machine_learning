#!/usr/bin/env python3
"""
Function that calculates the expectation step
in the EM algorithm for a GMM
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for
    each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster
    You may use at most 1 loop
    Returns: g, l, or None, None on failure
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    mix = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])  # Probabilidad del punto según la gaussiana i
        mix[i] = pi[i] * P      # Multiplicamos por el peso del clúster

    # Normalizar responsabilidades (para que sumen 1 en cada punto)
    g = mix / np.sum(mix, axis=0, keepdims=True)

    # Log-likelihood: suma del log de la mezcla total
    log_likelihood = np.sum(np.log(np.sum(mix, axis=0)))

    return g.T, log_likelihood
