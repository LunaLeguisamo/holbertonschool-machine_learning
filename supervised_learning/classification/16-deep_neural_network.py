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
        # Check that nx is an integer greater than 0
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Check that layers is a list of positive integers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        # Initialize the number of layers
        self.L = len(layers)

        # Initialize an empty dictionary to store intermediate values
        self.cache = {}

        # Initialize weights and biases
        self.weights = {}

        # Loop through each layer to initialize weights and biases
        for layer in range(1, self.L + 1):
            if layer == 1:
                prev_nodes = nx  # Number of inputs for the first layer
            else:
                prev_nodes = layers[layer - 2]

            # Initialize weights with He et al. method
            self.weights["W" + str(layer)] = (
                np.random.randn(layers[l - 1], prev_nodes)
                * np.sqrt(2 / prev_nodes)
                )

            # Initialize biases with zeros
            self.weights["b" + str(layer)] = np.zeros((layers[l - 1], 1))
