#!/usr/bin/env python3

"""
Function that add two arrays of the
the same shape
"""

import numpy as np


def add_arrays(arr1, arr2):
    """
    Si ambas matrices tienen igual forma,
    las sumamamos y agregamos las sumas
    a otra matriz arr3
    """

    if np.shape(arr1) == np.shape(arr2):
        arr3 = []
        for i in range(len(arr1)):
            arr3.append(arr1[i]+arr2[i])
        return arr3
    return None
