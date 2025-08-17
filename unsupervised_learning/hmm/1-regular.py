#!/usr/bin/env python3
"""
Write the function that determines the steady state
probabilities of a regular markov chain
"""

import numpy as np


def regular(P):
    """
    P is a square 2D numpy.ndarray representing the transition matrix.
    Returns the steady state probabilities or None on failure.
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

    # Solve the linear system: π(P - I) = 0, with ∑π = 1
    A = (P - np.eye(n)).T  # Transpose for solving Ax = 0
    b = np.zeros(n)
    # Replace last equation with ∑π = 1
    A[-1] = np.ones(n)
    b[-1] = 1

    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None  # Singular matrix (shouldn't happen for regular chains)

    # Round to 8 decimal places and reshape to (1, n)
    pi_rounded = np.round(pi, 8).reshape(1, -1)
    return pi_rounded
