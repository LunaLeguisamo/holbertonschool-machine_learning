#!/usr/bin/env python3
"""
Calcula la matriz transpuesta de una matriz dada.
"""


def matrix_transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        new_matrix.append(row)
    return new_matrix
