#!/usr/bin/env python3

"""
"""


def mat_mul(mat1, mat2):
    """
    """

    if len(mat1[0]) == len(mat2):
        mat3 =[]
        mat4 = []
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                mat3 = mat1[i] * mat2[0][j]
            mat4.append(mat3)
        return mat4
    return None