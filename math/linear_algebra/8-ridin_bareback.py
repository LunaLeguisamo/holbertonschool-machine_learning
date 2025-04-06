#!/usr/bin/env python3

"""
Function that multiplicate a 2D matrix
"""


def mat_mul(mat1, mat2):
    """
    We use three loops to multiplicte
    the matrixs
    """

    mat3 = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            column_j = [mat2[k][j] for k in range(len(mat2))]
            dot_prod = 0
            row_i = mat1[i]
            for k in range(len(mat1[0])):
                dot_prod += row_i[k] * column_j[k]
            row.append(dot_prod)
        mat3.append(row)
    return mat3
