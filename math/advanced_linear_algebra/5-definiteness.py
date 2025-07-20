#!/usr/bin/env python3
"""
Function that calculates the definiteness of a matrix
"""

import numpy as np


def definiteness(matrix):
    """
    matrix is a numpy.ndarray of shape (n, n) whose definiteness should be
    calculated
    If matrix is not a numpy.ndarray, raise a TypeError with the message matrix
    must be a numpy.ndarray
    If matrix is not a valid matrix, return None
    Return: the string Positive definite, Positive semi-definite, Negative
    semi-definite, Negative definite, or Indefinite if the matrix is positive
    definite, positive semi-definite, negative semi-definite, negative definite
    of indefinite, respectively
    If matrix does not fit any of the above categories, return None
    """
    # 1. Validación de tipo
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # 2. Validación de forma cuadrada
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    n = matrix.shape[0]

    # 3. Verificar que la matriz sea simétrica
    if not np.allclose(matrix, matrix.T):
        return None

    # 4. Determinar menores principales y verificar sus signos
    eigenvalues = np.linalg.eigvalsh(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
