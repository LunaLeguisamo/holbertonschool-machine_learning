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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    clss = np.zeros(X.shape[0], dtype=int)

    for _ in range(iterations):
        X_vectors = np.repeat(X[:, np.newaxis], k, axis=1)
        X_vectors = np.reshape(X_vectors, (X.shape[0], k, X.shape[1]))
        C_vectors = np.tile(C[np.newaxis, :], (X.shape[0], 1, 1))
        C_vectors = np.reshape(C_vectors, (X.shape[0], k, X.shape[1]))
        # Calculate Euclidean distances
        distances = np.linalg.norm(X_vectors - C_vectors, axis=2)
        new_clss = np.argmin(distances, axis=1)

        C_prev = C.copy()
        for j in range(k):
            mask = (new_clss == j)
            if np.any(mask):
                C[j] = X[mask].mean(axis=0)
            else:
                C[j] = np.random.uniform(
                    low=min_vals, high=max_vals, size=X.shape[1])

        if np.all(C == C_prev):
            return C, clss
        C_vectors = np.tile(C, (X.shape[0], 1))
        C_vectors = C_vectors.reshape(X.shape[0], k, X.shape[1])
        distance = np.linalg.norm(X_vectors - C_vectors, axis=2)
        clss = np.argmin(distance ** 2, axis=1)

    return C, clss
