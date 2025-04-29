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
        A (float): Activated output of the neuron (prediction), initially 0.
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
            - W is initialized with random normal values with shape (1, nx),
              meaning 1 neuron, nx input features.
            - b (bias) is initialized at 0.
            - A is the activated output and starts at 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Weight vector of the neuron.

        Returns:
            numpy.ndarray: Shape (1, nx), the weights of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Bias of the neuron.

        Returns:
            float: The bias, initialized to 0.
        """
        return self.__b

    @property
    def A(self):
        """
        Activated output (prediction) of the neuron.

        Returns:
            float: The activation value after forward propagation.
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X (ndarray): The input data, shape (nx, m), where nx is the
                         number of input features and m is the number of
                         examples.

        Updates:
            __A (float): The activated output (prediction) of the neuron.

        Returns:
            float: The activated output after forward propagation, which is
                   the prediction of the neuron.
        """
        # Compute the linear combination of inputs, weights, and bias
        z = (self.__W @ X) + self.b

        # Apply sigmoid activation function
        activation = 1 / (1 + np.exp(-z))

        # Store the activated output
        self.__A = activation

        return self.__A
