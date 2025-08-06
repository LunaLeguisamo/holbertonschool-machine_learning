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
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    # Check if rows sum to 1
    if not np.allclose(np.sum(P, axis=1), 1):
        return None

    # Check if regular: P^k > 0 for some k
    n = P.shape[0]
    power = np.copy(P)
    for _ in range(1, 100):
        power = power @ P
        if np.all(power > 0):
            break
    else:
        return None  # Not regular

    # Use initial distribution
    pi = np.ones((1, n)) / n
    for _ in range(1000):
        pi_next = pi @ P
        if np.allclose(pi_next, pi, atol=1e-8):
            return np.round(pi_next, 8)
        pi = pi_next

    return np.round(pi, 8)
