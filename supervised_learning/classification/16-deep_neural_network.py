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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(1, self.L):
            if not isinstance(layers[i], int) and layers[i] > 0:
                raise TypeError("layers must be a list of positive integers")
            else:
                if i == 1:
                    prev_nodes = nx
                else:
                    prev_nodes = layers[i - 2]

                self.weights["W" + str(i)] = (
                    np.random.randn(layers[i - 1], prev_nodes)
                    * np.sqrt(2 / prev_nodes)
                    )

                self.weights["b" + str(i)] = np.zeros((layers[i - 1], 1))
