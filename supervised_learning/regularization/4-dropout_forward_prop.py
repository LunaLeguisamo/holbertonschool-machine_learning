#!/usr/bin/env python3
"""Dropout forward propagation"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Performs forward propagation with Dropout.

    Args:
        X (numpy.ndarray): input data of shape (nx, m)
        weights (dict): weights and biases of the neural network
        L (int): number of layers
        keep_prob (float): probability of keeping a neuron active

    Returns:
        dict: cache containing all intermediary values of the network
    """
    cache = {"A0": X}

    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        A_prev = cache["A" + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i != L:
            # Activación con tanh
            A = np.tanh(Z)

            # Dropout mask
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A *= D
            A /= keep_prob

            # Guardamos A y D en cache
            cache["D" + str(i)] = D
            cache["A" + str(i)] = A
        else:
            # Última capa: softmax
            e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = e_Z / np.sum(e_Z, axis=0, keepdims=True)
            cache["A" + str(i)] = A

    return cache
