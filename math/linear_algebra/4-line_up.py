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
    if not matrix:
        return []

    if not isinstance(matrix[0], list):
        return [len(matrix)]

    return [len(matrix)] + matrix_shape(matrix[0])


"""
Function that add two arrays of the
the same shape
"""


def add_arrays(arr1, arr2):
    """
    Si ambas matrices tienen igual forma,
    las sumamamos y agregamos las sumas
    a otra matriz arr3
    """

    if matrix_shape(arr1) == matrix_shape(arr2):
        arr3 = []
        for i in range(len(arr1)):
            arr3.append(arr1[i]+arr2[i])
        return arr3
    return None
