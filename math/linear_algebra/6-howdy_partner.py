#!/usr/bin/env python3

"""
A function that concatenate two arrays in other
"""


def cat_arrays(arr1, arr2):
    """
    For concatenate two arrays, we use
    two loops with the len of the arrays
    to concatenate
    """

    arr3 = []
    for i in range(len(arr1)):
        arr3.append(arr1[i])
    for i in range(len(arr2)):
        arr3.append(arr2[i])
    return arr3
