#!/usr/bin/env python3
"""
Function that updates the weights and biases of a neural network
using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization
    """
    m = Y.shape[1]
    weights_cpy = weights.copy()

    for i in reversed(range(1, L + 1)):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights_cpy['W' + str(i)]

        if i == L:
            dZ = A - Y
        else:
            dZ = dZ_prev * (1 - A ** 2)

        dW = np.dot(dZ, A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            dZ_prev = np.dot(W.T, dZ)
