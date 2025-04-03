#!/usr/bin/env python3
"""
Calcula la forma (dimensiones) de una matriz multidimensional.
"""


def matrix_shape(matrix):
    """
    Calcula la forma (dimensiones) de una matriz multidimensional.

    La función usa recursividad para calcular la forma de la matriz.
    Si la matriz tiene más de una dimensión,
    la función se llama a sí misma para explorar cada nivel de profundidad
    y obtener el número de elementos en
    cada nivel.
    """

    if not isinstance(matrix[0], list):
        return [len(matrix)]

    return [len(matrix)] + matrix_shape(matrix[0])
