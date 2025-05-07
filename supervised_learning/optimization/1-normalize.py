#!/usr/bin/env python3
"""
Function that normalizes
(standardizes) a matrix
"""

import numpy as np


def normalize(X, m, s):
    """
    We apply the formula to normalize
    a matrix with his original entry point(X),
    his mean (m) and standard deviation (s)
    """
    n = (X - m) / s
    return n
