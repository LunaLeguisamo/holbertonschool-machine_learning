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

    # Usamos log-probabilities para evitar underflow
    log_V = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Paso inicial (en log)
    log_V[:, 0] = np.log(Initial[:, 0]) + np.log(Emission[:, Observation[0]])

    # Iteración Viterbi
    for t in range(1, T):
        for j in range(N):
            log_prob = log_V[:, t-1] + np.log(Transition[:, j])\
                + np.log(Emission[j, Observation[t]])
            log_V[j, t] = np.max(log_prob)
            backpointer[j, t] = np.argmax(log_prob)

    # Probabilidad total (en log)
    log_P = np.max(log_V[:, -1])
    P = np.exp(log_P)  # Convertimos a probabilidad normal (opcional)

    # Reconstrucción del camino (backtracking)
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(log_V[:, -1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    return path.tolist(), P
