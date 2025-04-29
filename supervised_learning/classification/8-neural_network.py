#!/usr/bin/env python3
"""
NeuralNetwork class for a binary classification problem using one hidden layer.
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary
    classification.
    """

    def __init__(self, nx, nodes):
        """
        Class constructor.

        Parameters:
        nx (int): Number of input features.
        nodes (int): Number of nodes in the hidden layer.

        Attributes initialized:
        W1 (ndarray): Weights for the hidden layer, shape (nodes, nx).
        b1 (ndarray): Biases for the hidden layer, shape (nodes, 1),
        initialized to zeros.
        A1 (float): Activated output of the hidden layer, initialized to 0.
        W2 (ndarray): Weights for the output layer, shape (1, nodes).
        b2 (float): Bias for the output neuron, initialized to 0.
        A2 (float): Activated output of the output neuron (prediction),
        initialized to 0.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        TypeError: If nodes is not an integer.
        ValueError: If nodes is less than 1.
        """
        # Validación de nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validación de nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Inicialización de los pesos y bias de la capa oculta
        self.W1 = np.random.randn(nodes, nx)  # distribución uniformealeatoria
        self.b1 = np.zeros((nodes, 1))       # bias inicializado en ceros
        self.A1 = 0                          # activación inicial

        # Inicialización de los pesos y bias de la neurona de salida
        self.W2 = np.random.randn(1, nodes)   # pesos para la capa de salida
        self.b2 = 0                          # bias de salida
        self.A2 = 0                          # activación de salida
