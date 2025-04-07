#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Esta función genera un gráfico que muestra el decaimiento
# exponencial del C-14
# a lo largo del tiempo, utilizando una escala logarítmica en
# el eje y.


def change_scale():
    # Creamos un array de valores para el tiempo (en años) con
    # un rango de 0 a 28,650.
    # Usamos un intervalo de 5,730 años (vida media del C-14).
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.xlim(0, 28650)
    plt.yscale('log')
    plt.plot(x, y)
    plt.show()
