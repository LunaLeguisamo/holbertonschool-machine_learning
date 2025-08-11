#!/usr/bin/env python3
"""
Function that performs the Baum-Welch
algorithm for a hidden markov model
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Observations is a numpy.ndarray of shape (T,) that contains the index of
    the observation
    T is the number of observations
    Transition is a numpy.ndarray of shape (M, M) that contains the initialized
    transition probabilities
    M is the number of hidden states
    Emission is a numpy.ndarray of shape (M, N) that contains the initialized
    emission probabilities
    N is the number of output states
    Initial is a numpy.ndarray of shape (M, 1) that contains the initialized
    starting probabilities
    iterations is the number of times expectation-maximization should be
    performed
    Returns: the converged Transition, Emission, or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    if Initial.shape[1] != 1:
        return None, None

    M = Transition.shape[0]
    N = Emission.shape[1]
    T = Observations.shape[0]

    for _ in range(iterations):
        # Forward-Backward para calcular alpha y beta
        alpha = np.zeros((M, T))
        alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]

        for t in range(1, T):
            for j in range(M):
                alpha[j, t] = np.sum(alpha[:, t-1] * Transition[:, j])\
                    * Emission[j, Observations[t]]

        # Backward
        beta = np.zeros((M, T))
        beta[:, -1] = 1

        for t in range(T-2, -1, -1):
            for i in range(M):
                beta[i, t] = np.sum(Transition[i, :] *
                                    Emission[:, Observations[t+1]] * beta[:,
                                                                          t+1])

        # Calcular gamma y xi
        gamma = np.zeros((M, T))
        xi = np.zeros((M, M, T-1))

        P = np.sum(alpha[:, -1])
        # Probabilidad total

        for t in range(T-1):
            for i in range(M):
                gamma[i, t] = alpha[i, t] * beta[i, t] / P
                for j in range(M):
                    xi[i, j, t] = alpha[i, t] * Transition[i, j]\
                        * Emission[j, Observations[t+1]] * beta[j, t+1] / P

        gamma[:, -1] = alpha[:, -1] * beta[:, -1] / P

        # Actualización de parámetros (Maximization)
        # Actualizar Transition
        Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1,
                                                 keepdims=True)

        # Actualizar Emission
        for k in range(N):
            mask = (Observations == k)
            Emission[:, k] = np.sum(gamma[:, mask], axis=1) / np.sum(gamma,
                                                                     axis=1)

    return Transition, Emission
