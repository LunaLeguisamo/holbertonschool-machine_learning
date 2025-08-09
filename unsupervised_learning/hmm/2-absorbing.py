#!/usr/bin/env python3
"""
Function that determines if a markov chain is absorbing
"""

import numpy as np


def absorbing(P):
    """
    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the standard transition matrix
    P[i, j] is the probability of transitioning
    from state i to state j
    n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    
    if P.shape[0] != P.shape[1]:
        return None
    
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    
    n, _ = P.shape
    
    for i in range(n):
        for j in range(n):
            if P[i][j] == 1 and 
    