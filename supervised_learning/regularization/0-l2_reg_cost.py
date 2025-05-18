#!/usr/bin/env python3
"""
Function that calculates the cost of a
neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    cost is the cost of the network without L2 regularization
    lambtha is the regularization parameter
    weights is a dictionary of the weights and biases (numpy.ndarrays)
    of the neural network
    L is the number of layers in the neural network
    m is the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    l2_sum = 0
    for i in range(1, L + 1):  # Claves van desde W1 hasta WL
        key = f'W{i}'
        l2_sum += np.sum(weights[key] ** 2)

    l2_term = (lambtha / (2 * m)) * l2_sum
    return cost + l2_term
