#!/usr/bin/env python3
"""
Function that calculates the most likely
sequence of hidden states for a hidden markov model
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Observation is a numpy.ndarray of shape (T,) that contains
    the index of the observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the
    emission probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the
    hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
    Transition[i, j] is the probability of transitioning from the
    hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state
    Returns: path, P, or None, None on failure
    path is the a list of length T containing the most likely sequence
    of hidden states
    P is the probability of obtaining the path sequence
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or Initial.shape[1] != 1:
        return None, None

    T = Observation.shape[0]
    N = Emission.shape[0]
    V = np.zeros((N, T))
    path = np.zeros(T, dtype=int)

    # Paso inicial
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Iteración Viterbi
    for t in range(1, T):
        for j in range(N):
            prob = V[:, t-1] * Transition[:, j] * Emission[j, Observation[t]]
            V[j, t] = np.max(prob)
            path[t] = np.argmax(prob)

    # Probabilidad total de la secuencia observada
    P = np.max(V[:, -1])

    # Reconstruir el camino
    for t in range(T - 1, 0, -1):
        path[t] = path[t - 1]

    return path.tolist(), P
