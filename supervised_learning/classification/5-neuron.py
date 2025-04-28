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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (ndarray): Shape (1, m), correct labels for the input data.
                         Each value must be 0 or 1.
            A (ndarray): Shape (1, m), activated output of the neuron
                         for each example (predictions).

        Returns:
            float: The logistic regression cost.

        Notes:
            - This function implements the cross-entropy loss.
            - The formula used is:
                cost = -(1/m) * sum(Y * log(A) + (1 - Y) * log(1 - A))
            - To prevent log(0), we use (1.0000001 - A) instead of (1 - A).
        """
        m = Y.shape[1]
        cost = np.sum(-(Y * np.log(A) + (1-Y) * np.log(1.0000001-A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions.

        Parameters:
            X (numpy.ndarray): Shape (nx, m) containing the input data.
                nx: number of input features.
                m: number of examples.
            Y (numpy.ndarray): Shape (1, m) containing the correct
            labels for the input data.

        Returns:
            tuple: predicted labels and the cost.
                - prediction (numpy.ndarray): shape (1, m) with predicted
                labels (1 if A >= 0.5, else 0).
                - cost (float): cost of the predictions using logistic
                regression loss.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Parameters:
            X (numpy.ndarray): Shape (nx, m) containing the input data.
                - nx is the number of input features.
                - m is the number of examples.
            Y (numpy.ndarray): Shape (1, m) containing the correct labels
            for the input data. A (numpy.ndarray): Shape (1, m) containing the
            activated output of the neuron for each example.

            alpha (float): Learning rate used to update the weights and bias
            (default is 0.05).

        Updates:
            __W (numpy.ndarray): Updated weights after one step of gradient
            descent.
            __b (float): Updated bias after one step of gradient descent.
        """
        m = X.shape[1]
        dz = A - Y
        dw = 1/m * dz @ X.T
        db = 1/m * np.sum(dz)
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
            