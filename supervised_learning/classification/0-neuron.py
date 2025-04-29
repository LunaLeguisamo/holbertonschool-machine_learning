#!/usr/bin/env python3
"""
This module defines a simple Neuron class for binary classification
using logistic regression.
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron performing binary classification.

    Attributes:
        nx (int): Number of input features.
        W (ndarray): Weights for the neuron, shape (1, nx).
        b (float): Bias initialized to 0.
        A (float): Activated output of the neuron (prediction),
        initially 0.
    """

    def __init__(self, nx):
        """
        Initializes the neuron.

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.

        Notes:
            - W is initialized with random normal values with shape
            (1, nx) because
              each input feature needs its own weight.
            - The output of a single neuron is a scalar (1 value),
              so W must be a row vector (1 row, nx columns).
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
