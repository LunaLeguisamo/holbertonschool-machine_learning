#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

"""
The x-axis should be labeled Time (years)
The y-axis should be labeled Fraction Remaining
The title should be Exponential Decay of Radioactive Elements
The x-axis should range from 0 to 20,000
The y-axis should range from 0 to 1
x ↦ y1 should be plotted with a dashed red line
x ↦ y2 should be plotted with a solid green line
A legend labeling x ↦ y1 as C-14 and x ↦ y2 as Ra-226 should
be placed in the upper right hand corner of the plot
"""


def two():
    """
    Esta función grafica la descomposición exponencial de dos
    elementos radiactivos (C-14 y Ra-226)
    a lo largo del tiempo. La función calcula y grafica la fracción
    restante de ambos elementos, considerando dos tiempos de vida diferentes:
    5730 años para el C-14 y 1600 años para el Ra-226.
    El gráfico muestra las dos curvas con estilos de línea diferentes y
    las etiquetas correspondientes en el eje x (Tiempo en años) y en el eje y
    (Fracción restante).
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")
    plt.plot(x, y1, label="C-14", color='red', linestyle='--')
    plt.plot(x, y2, label="Ra-226", c='green')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
