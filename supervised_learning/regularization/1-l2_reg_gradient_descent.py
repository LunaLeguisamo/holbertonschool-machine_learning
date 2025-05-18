#!/usr/bin/env python3
"""
Function  that updates the weights
and biases of a neural network using
gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains
    the correct labels for the data
    classes is the number of classes
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs of each layer of the neural network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]

    for i in reversed(range(1, L + 1)):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        if i == L:
            dZ = A - Y
        else:
            #  derivada tanh
            dZ = dZ_prev * A * (1 - A)

        if i > 1:
            dZ_prev = np.dot(W.T, dZ)

        dW = np.dot(dZ, A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
