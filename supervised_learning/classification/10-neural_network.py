#!/usr/bin/env python3
"""
Clase NeuralNetwork para resolver un problema de clasificación binaria
usando una red neuronal con una capa oculta.
"""

import numpy as np


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
