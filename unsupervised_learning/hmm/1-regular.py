#!/usr/bin/env python3
"""
Write the function that determines the
steady state probabilities of a regular markov chain
"""

import numpy as np


def regular(P):
    """
    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None

    if P.shape[0] != P.shape[1]:
        return None
    
    if not np.allclose(P.sum(axis=1), 1):
        return None
    
    is_regular = False
    for i in range(1, 20):
        P = np.linalg.matrix_power(P, i)
        if np.all(P > 0):
            is_regular = True
            break

    if not is_regular:
        return None

    n, _ = P.shape

    s = np.ones((1, n)) / n  # distribución uniforme inicial

    for i in range(20):
        s_next = s @ P
        if np.allclose(s_next, s, atol=1e-8):
            break
        s = s_next

    return np.round(s, 8)
