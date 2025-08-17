#!/usr/bin/env python3
"""
Function that performs the
forward algorithm for a hidden markov model
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Observation is a numpy.ndarray of shape (T,) that contains the index
    of the observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities Transition[i, j] is the probability of transitioning from the
    hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state
    Returns: P, F, or None, None on failure
    P is the likelihood of the observations given the model
    F is a numpy.ndarray of shape (N, T) containing the forward path
    probabilities
    F[i, j] is the probability of being in hidden state i at time j given
    the previous observations
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
    F = np.zeros((N, T))

    # Paso inicial
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Iteración Forward
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t-1] * Transition[:, j])\
                * Emission[j, Observation[t]]

    # Probabilidad total de la secuencia observada
    P = np.sum(F[:, -1])

    return P, F
