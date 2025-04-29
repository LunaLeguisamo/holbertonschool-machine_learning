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
        if not isinstance(layers, list) or len(layers) == 0 or not all(
            isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_nodes = nx

        for i in range(self.L):
            self.__weights['W' + str(i + 1)] = (
                np.random.randn(layers[i], prev_nodes)
                * np.sqrt(2 / prev_nodes)
            )
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
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
        self.__cache['A0'] = X

        for i in range(1, self.L + 1):
            Z = np.dot(self.__weights['W' + str(i)], self.__cache['A' + str(i - 1)]) + self.__weights['b' + str(i)]
            self.__cache['A' + str(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache['A' + str(self.L)], self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        L = self.L
        dz = 0

        for i in reversed(range(1, L + 1)):
            A = cache['A' + str(i)]
            A_prev = cache['A' + str(i - 1)]

            if i == L:
                dz = A - Y
            else:
                dz = np.dot(self.weights['W' + str(i + 1)].T, dz) * A * (1 - A)

            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            self.weights['W' + str(i)] -= alpha * dw
            self.weights['b' + str(i)] -= alpha * db
