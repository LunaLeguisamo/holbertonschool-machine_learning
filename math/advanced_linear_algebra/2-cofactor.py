#!/usr/bin/env python3
"""
Function that calculates the cofactor matrix of a matrix
"""


def minor(matrix):
    """
    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message matrix
    must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix
    Returns: the minor matrix of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if matrix == [[]]:
        return [[1]]

    size = len(matrix)
    if size == 1:
        return [[1]]

    minors = []

    for i in range(size):  # fila
        row_minors = []
        for j in range(size):  # columna
            # submatriz sin fila i y columna j
            submatrix =\
                [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]
            row_minors.append(determinant(submatrix))
        minors.append(row_minors)

    return minors


def cofactor(matrix):
    """
    matrix is a list of lists whose cofactor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message matrix
    must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix
    Returns: the cofactor matrix of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor = []
    size = len(matrix)
    minors = minor(matrix)

    for i in range(size):  # fila
        rows = []
        for j in range(size):  # columna
            rows.append((-1) ** (i+j) * minors[i][j])
        cofactor.append(rows)

    return cofactor
