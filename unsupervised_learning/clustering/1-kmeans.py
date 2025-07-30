#!/usr/bin/env python3
"""
Function that performs K-means on a dataset
"""
import numpy as np


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate uniform
    distribution along each dimension in d:
    The minimum values for the distribution should be the minimum values of X
    along each dimension in d
    The maximum values for the distribution should be the maximum values of X
    along each dimension in d
    You should use numpy.random.uniform exactly once
    You are not allowed to use any loops
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    centroids = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(k, d))

    return centroids


def kmeans(X, k, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed
    If no change in the cluster centroids occurs between iterations,
    your function should return
    Initialize the cluster centroids using a multivariate uniform distribution
    If a cluster contains no data points during the update step, reinitialize
    its centroid
    You should use numpy.random.uniform exactly twice
    You may use at most 2 loops
    Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster clss is a numpy.ndarray of shape (n,) containing the index of
    the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):  # ← ÚNICO bucle permitido
        # Asignación de puntos a centroides (sin bucle)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)  # (n, k)
        clss = np.argmin(distances, axis=1)  # (n,)

        # Actualización de centroides (también sin bucle)
        C_new = np.array([
            X[clss == i].mean(axis=0) if np.any(clss == i)
            else np.random.uniform(X.min(axis=0), X.max(axis=0))
            for i in range(k)
        ])

        if np.allclose(C, C_new):
            break
        C = C_new

    return C, clss
