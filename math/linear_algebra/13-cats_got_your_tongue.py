#!/usr/bin/env python3

"""
This function concatenates two
numpy arrays along a specified axis
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    This function concatenates
    two numpy arrays along a specified axis
    """

    return np.concatenate((mat1, mat2), axis=axis)
