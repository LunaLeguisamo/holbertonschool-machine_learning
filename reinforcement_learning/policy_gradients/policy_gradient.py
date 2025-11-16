#!/usr/bin/env python3
"""
Funcion that computes the policy with a weight of a matrix.
"""
import numpy as np


def policy(matrix, weight):
    """Calculates the action to take using a policy gradient

    Args:
        matrix (numpy.ndarray): 2D array representing the state
        weight (numpy.ndarray): 2D array representing the weights

    Returns:
        numpy.ndarray: 1D array of action probabilities
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))  # for numerical stability
    return exp / exp.sum(axis=1, keepdims=True)
