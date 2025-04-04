#!/usr/bin/env python3

"""
Function that concatenate two matrix 2D
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    We use four conditions and one loop
    to obtain other matrix
    """
    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            mat3 = mat1+mat2
            return mat3

    if axis == 1:
        if len(mat1) == len(mat2):
            mat3 = []
            for i in range(len(mat1)):
                mat3.append(mat1[i]+mat2[i])
            return mat3
    return None
