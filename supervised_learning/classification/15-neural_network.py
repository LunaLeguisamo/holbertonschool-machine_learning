#!/usr/bin/env python3
"""
Clase NeuralNetwork para resolver un problema de clasificación binaria
usando una red neuronal con una capa oculta.
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Clase que define una red neuronal con una capa oculta,
    utilizada para clasificación binaria.
    """

    def __init__(self, nx, nodes):
        """
        Constructor de la clase.

        Parámetros:
        nx (int): Cantidad de características de entrada (features).
        nodes (int): Número de nodos en la capa oculta.

        Atributos públicos inicializados:
        W1 (ndarray): Pesos de la capa oculta, con forma (nodes, nx),
                      inicializados con una distribución normal.
        b1 (ndarray): Bias de la capa oculta, con forma (nodes, 1),
                      inicializado en 0.
        A1 (float): Activación de la capa oculta, inicializada en 0.
        W2 (ndarray): Pesos de la capa de salida, con forma (1, nodes),
                      inicializados con una distribución normal.
        b2 (float): Bias de la neurona de salida, inicializado en 0.
        A2 (float): Activación de la neurona de salida, inicializada en 0.

        Excepciones:
        TypeError: Si nx no es un entero.
        ValueError: Si nx es menor a 1.
        TypeError: Si nodes no es un entero.
        ValueError: Si nodes es menor a 1.
        """
        # Validación del número de entradas
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validación del número de nodos
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Inicialización de parámetros
        self.__W1 = np.random.randn(nodes, nx)   # Pesos de la capa oculta
        self.__b1 = np.zeros((nodes, 1))         # Bias de la capa oculta
        self.__A1 = 0                            # Activación hiddenlayer
        self.__W2 = np.random.randn(1, nodes)    # Pesos de la capa de salida
        self.__b2 = 0                            # Bias de salida
        self.__A2 = 0                            # Activación de salida

    @property
    def W1(self):
        """Devuelve los pesos de la capa oculta."""
        return self.__W1

    @property
    def b1(self):
        """Devuelve el bias de la capa oculta."""
        return self.__b1

    @property
    def A1(self):
        """Devuelve la activación de la capa oculta."""
        return self.__A1

    @property
    def W2(self):
        """Devuelve los pesos de la neurona de salida."""
        return self.__W2

    @property
    def b2(self):
        """Devuelve el bias de la neurona de salida."""
        return self.__b2

    @property
    def A2(self):
        """Devuelve la activación de la neurona de salida."""
        return self.__A2

    def forward_prop(self, X):
        """
        Realiza la propagación hacia adelante de la red neuronal.

        Parámetros:
        X (ndarray): Input de datos con forma (nx, m), donde:
                    - nx es el número de características
                    - m es el número de ejemplos

        Retorna:
        La activación de la capa oculta (A1) y la activación de salida (A2)
        """
        # Propagación en la capa oculta
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))  # función sigmoide

        # Propagación en la capa de salida
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

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
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        pred = np.where(A2 >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one step of gradient descent to update the weights
        and biases of the neural network using the gradients computed
        during backpropagation.

        Parameters:
        X (ndarray): Input data with shape (nx, m), where nx is the
        number of feature and m is the number of examples.
        Y (ndarray): True labels for the input data, with shape (1, m).
        A1 (ndarray): The activation values of the first layer (hidden layer),
        with shape (n1, m).
        A2 (ndarray): The activation values of the second layer (output layer),
        with shape (1, m).
        alpha (float): The learning rate for gradient descent.

        Updates:
        - The weights (`__W1`, `__W2`) and biases (`__b1`, `__b2`) are updated
        in place using the gradient descent rule, based on the computed
        gradients.

        Example:
        nn.gradient_descent(X_train, Y_train, A1, A2, alpha=0.01)
        """
        m = X.shape[1]
        dz2 = A2 - Y
        dz1 = (self.__W2.T @ dz2) * (A1 * (1 - A1))
        dw1 = 1/m * dz1 @ X.T
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)
        dw2 = 1/m * dz2 @ A1.T
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neural network using gradient descent over a
        specified number of iterations.
        It also computes the cost after each iteration, optionally
        displaying or plotting the training progress.

        Parameters:
        X (ndarray): Input data with shape (nx, m), where nx is the
        number of features an m is the number of examples.
        Y (ndarray): True labels for the input data, with shape (1, m).
        iterations (int): The number of iterations for gradient descent.
        Default is 5000.
        alpha (float): The learning rate for gradient descent. Default is 0.05.
        verbose (bool): Whether to print the cost after each `step` iterations.
        Default is True.
        graph (bool): Whether to plot the cost over iterations. Default is True.
        step (int): The number of iterations between printing the cost
        (if `verbose` is True).
        Default is 100.

        Returns:
        pred (ndarray): The predicted labels after training.
        cost (float): The final cost after training.

        Raises:
        TypeError: If `iterations` is not an integer, `alpha` is not a float,
        or `step` is not
        an integer.
        ValueError: If `iterations` is not a positive integer, `alpha` is not
        positive, or `step`
        is not positive or greater than zero and less than or equal to
        `iterations`.

        Process:
        1. Validate input arguments for type and range.
        2. Initialize a list to store the cost at each iteration.
        3. Print the cost after every `step` iterations if `verbose` is True.
        4. Optionally plot the cost as a function of iteration if `graph`
        is True.
        5. Return the predicted labels (`pred`) and final cost (`cost`).
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
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            pred, cost = self.evaluate(X, Y)
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")
                iteration.append(i)
                costs.append(self.cost(Y, self.__A2))

        if graph:
            plt.plot(iteration, costs, color="blue")
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return pred, cost
