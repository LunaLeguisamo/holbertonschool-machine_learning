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
Function that add two 2D matrix of the
the same shape
"""


def add_matrices2D(mat1, mat2):
    """
    Comparo si dos matrices 2D tienen la misma forma,
    luego creo otra matrix vacia donde agregare dos arrays,
    creo un array mat3 fuera del bucle, y
    """
    if matrix_shape(mat1) == matrix_shape(mat2):
        mat3 = []
        for i in range(len(mat1)):
            mat4 = []
            for j in range(len(mat1[0])):
                mat4.append(mat1[i][j]+mat2[i][j])
            mat3.append(mat4)
        return mat3
    return None
