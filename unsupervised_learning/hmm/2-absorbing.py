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

    n = P.shape[0]
    absorbent_states = [i for i in range(n) if np.isclose(P[i, i], 1.0)]

    if not absorbent_states:
        return False

    non_absorbent = [i for i in range(n) if i not in absorbent_states]

    for state in non_absorbent:
        visited = set()
        queue = [state]
        found = False

        while queue and not found:
            current = queue.pop(0)
            if current in absorbent_states:
                found = True
                break
            if current in visited:
                continue
            visited.add(current)
            # Añadir todos los estados a los que puede transicionar
            for next_state in range(n):
                if P[current, next_state] > 0:
                    queue.append(next_state)

        if not found:
            return False

    return True
