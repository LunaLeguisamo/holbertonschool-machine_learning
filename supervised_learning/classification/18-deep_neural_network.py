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
            if not isinstance(layers[i], int) or layers[i] < 0:
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
