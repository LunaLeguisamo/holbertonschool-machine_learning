#!/usr/bin/env python3
"""
This module provides a function to compute
the normalization constants of a given data
matrix X.

Normalization (or standardization) helps
ensure that each feature contributes
equally to model training, especially in
gradient-based methods.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation
    for each feature in X.

    Parameters:
    -----------
    X : numpy.ndarray of shape (m, nx)
        The input data matrix where:
        - m is the number of data points (samples)
        - nx is the number of features per data point

    Returns:
    --------
    mean : numpy.ndarray of shape (nx,)
        The mean value of each feature (column-wise average)
    std : numpy.ndarray of shape (nx,)
        The standard deviation of each feature
        (column-wise std deviation)

    Notes:
    ------
    These statistics are commonly used for data normalization:
        normalized_X = (X - mean) / std
    This ensures that each feature has zero mean and unit variance,
    which accelerates convergence in many optimization algorithms.
    """

    # Calculamos la media de cada columna (feature) de la matriz X
    mean = np.mean(X, axis=0)

    # Calculamos la desviación estándar de cada columna
    # (feature) de la matriz X
    std = np.std(X, axis=0)

    # Devolvemos ambos vectores: uno con las medias y otro
    # con las desviaciones estándar
    return mean, std
