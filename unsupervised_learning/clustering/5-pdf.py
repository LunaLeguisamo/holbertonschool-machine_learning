#!/usr/bin/env python3
"""
Function that calculates the probability density function of a
Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data
    points whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of
    the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance
    of the distribution
    You are not allowed to use any loops
    You are not allowed to use the function numpy.diag or the method
    numpy.ndarray.diagonal
    Returns: P, or None on failure
    P is a numpy.ndarray of shape (n,) containing the PDF values
    for each data point
    All values in P should have a minimum value of 1e-300
    """
    diff = X - m
    inv_S = np.linalg.inv(S)
    det_S = np.linalg.det(S)
    quad = np.einsum('ni,ij,nj->n', diff, inv_S, diff)
    const = 1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * det_S)
    P = const * np.exp(-0.5 * quad)
    P = np.maximum(P, 1e-300)
    return P
