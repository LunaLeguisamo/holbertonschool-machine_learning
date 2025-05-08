#!/usr/bin/env python3
"""
Function that creates mini-batches from two matrices X and Y
in a synchronized way, to be used in mini-batch gradient descent.
"""

import numpy as np


def moving_average(data, beta):
    """
    comment
    """
    v = 0
    fix_prom = []

    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        v_t = v / (1 - beta**(i+1))
        fix_prom.append(v_t)
    return fix_prom
