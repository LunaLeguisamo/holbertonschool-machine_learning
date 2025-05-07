#!/usr/bin/env python3
"""
Function that  that shuffles the
data points in two matrices the
same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Apply np.random.permutation to shuffle
    the matrix's
    """
    return np.random.permutation(X), np.random.permutation(Y)
