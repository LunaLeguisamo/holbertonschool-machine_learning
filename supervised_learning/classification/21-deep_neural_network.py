#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network for binary classification
    """
    def __init__(self, nx, layers):
        """
        Class constructor

        Parameters:
        - nx (int): Number of input features
        - layers (list): List representing the number of nodes in each layer

        Attributes:
        - L (int): Number of layers in the neural network
        - cache (dict): Holds all intermediary values of the network
        - weights (dict): Holds all weights and biases of the network

        Raises:
        - TypeError: If nx is not an integer
        - ValueError: If nx is less than 1
        - TypeError: If layers is not a list of positive integers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_nodes = nx

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            # Pesos inicializados con He initialization
            self.weights["W" + str(i + 1)] = (
                np.random.randn(layers[i], prev_nodes)
                * np.sqrt(2 / prev_nodes)
                )

            # Bias inicializados en ceros
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

            prev_nodes = layers[i]

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Parameters:
        - X (numpy.ndarray): Input data of shape (nx, m)

        Updates:
        - __cache: Stores the activations of each layer (including input X)

        Returns:
        - A (numpy.ndarray): The output of the neural network (activation
        from the last layer)
        - cache (dict): Dictionary containing all intermediary activations
        """
        self.__cache['A0'] = X

        for i in range(1, self.L + 1):
            z = np.dot(
                self.__weights['W' + str(i)], self.__cache['A' + str(i-1)]
                ) + self.__weights['b' + str(i)]
            self.__cache['A' + str(i)] = 1 / (1 + np.exp(-z))

        return self.__cache['A' + str(i)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        (cross-entropy loss).

        Parameters:
        Y (ndarray): Correct labels for the input data, shape (1, m).
        A (ndarray): Activated output (predictions) of the model for
        each example, shape (1, m).

        Returns:
        float: The cost (loss) computed using logistic regression.

        Notes:
        - The cost function used is the binary cross-entropy:
        cost = -(1/m) * Σ [Y * log(A) + (1 - Y) * log(1 - A)]
        - A small constant (1.0000001 instead of 1) is used inside
        log to avoid numerical errors
        like log(0), which would cause computational issues.
        """
        m = Y.shape[1]
        cost = np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the predictions of the neural network.

        Parameters:
        X (numpy.ndarray): Input data of shape (nx, m),
            where nx is the number of input features and m is
            the number of examples.
        Y (numpy.ndarray): Correct labels for the input data,
        of shape (1, m).

        Returns:
        tuple: (prediction, cost)
            - prediction (numpy.ndarray): Array of shape (1, m)
            containing the predicted labels
            (1 if the output activation is >= 0.5, 0 otherwise).
            - cost (float): Cost of the predictions compared to the
            correct labels.

        Process:
        - Performs forward propagation to calculate the activations.
        - Calculates the cost using the predicted activations and the
        true labels.
        - Generates predictions by thresholding the output activation at 0.5.
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        L = self.__L
        dz = 0

        for i in reversed(range(1, L + 1)):
            A = cache['A' + str(i)]
            A_prev = cache['A' + str(i - 1)]

            if i == L:
                dz = A - Y
            else:
                W_next = self.__weights['W' + str(i + 1)]
                dz = np.dot(W_next.T, dz) * A * (1 - A)

            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db
