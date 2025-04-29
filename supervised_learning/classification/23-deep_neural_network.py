#!/usr/bin/env python3
"""
Defines a deep neural network performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt

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
        """
        Performs one pass of gradient descent on the neural network

        Parameters:
        - Y (numpy.ndarray): Correct labels, shape (1, m)
        - cache (dict): Dictionary containing all intermediary values
        of the network
        - alpha (float): Learning rate
        """
        m = Y.shape[1]
        L = self.__L
        weights_copy = self.__weights.copy()

        for i in reversed(range(1, L + 1)):
            A = cache["A" + str(i)]
            A_prev = cache["A" + str(i - 1)]

            if i == L:
                dZ = A - Y
            else:
                W_next = weights_copy["W" + str(i + 1)]
                dZ = dA_prev * A * (1 - A)

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                dA_prev = np.dot(weights_copy["W" + str(i)].T, dZ)

            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network using gradient descent.

        Parameters:
        - X (numpy.ndarray): Input data of shape (nx, m), where
        nx is the number of features and m is the number of examples.
        - Y (numpy.ndarray): Correct labels for the input data,
        shape (1, m).
        - iterations (int): The number of iterations to train the model.
        Default is 5000.
        - alpha (float): The learning rate to be used in gradient descent.
        Default is 0.05.

        Returns:
        - tuple: (prediction, cost)
            - prediction (numpy.ndarray): Array of shape (1, m) containing
            the predicted labels.
            - cost (float): The cost of the predictions compared to the true
            labels.

        Process:
        - Validates input types and values for `iterations` and `alpha`.
        - Performs `iterations` number of forward propagation and
        backpropagation
        steps.
        - After each iteration, the model adjusts its weights using gradient
        descent.
        - After all iterations, it evaluates the model's performance
        (predictions and cost).

        Notes:
        - `forward_prop` calculates activations from the input data, storing
        intermediate values in `cache`.
        - `gradient_descent` updates the model's weights using the
        backpropagated
        gradients.
        - `evaluate` is used to calculate the final predictions and cost.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step > iterations or step <= 0:
                raise ValueError("step must be positive and <= iterations")

        pred, cost = self.evaluate(X, Y)
        costs = []
        iteration = [0]
        costs.append(cost)
        if verbose:
            print(f"Cost after 0 iterations: {cost}")

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            pred, cost = self.evaluate(X, Y)
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")
                iteration.append(i)
                costs.append(self.cost(Y, A))

        if graph:
            plt.plot(iteration, costs, color="blue")
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return pred, cost
