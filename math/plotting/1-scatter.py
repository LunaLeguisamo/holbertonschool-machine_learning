#!/usr/bin/env python3

"""
Esta función genera un gráfico de dispersión (scatter plot) de
alturas y pesos
de 2000 hombres utilizando una distribución normal bivariada.
Los puntos se
muestran en color magenta y el gráfico tiene etiquetas y título.
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Esta función genera un gráfico de dispersión (scatter plot) de alturas
    y pesos de 2000 hombres utilizando una distribución normal bivariada.
    Los puntos se muestran en color magenta y el gráfico tiene etiquetas y
    título.
    """

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.scatter(x, y, color='m')
    plt.show()
