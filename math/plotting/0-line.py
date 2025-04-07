#!/usr/bin/env python3

"""
Módulo: 0-line.py

Este módulo contiene una función que genera un gráfico de línea
representando la función cúbica y = x³ utilizando matplotlib.

Funciones:
    line(): Grafica una línea roja continua desde x=0 hasta x=10.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Genera un gráfico de línea para la función y = x³.

    - El eje x va de 0 a 10.
    - El eje y muestra el cubo de cada valor de x.
    - El gráfico se presenta como una línea roja sólida.
    - Se utiliza un tamaño de figura estándar de 6.4x4.8 pulgadas.

    Muestra el gráfico en una ventana emergente.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 10)
    plt.plot(range(11), y, 'r')
    plt.show()
