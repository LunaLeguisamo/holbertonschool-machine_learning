#!/usr/bin/env python3

"""
summ
"""


def summation_i_squared(n):
    if n > 0 and isinstance(n, int):
        a = (n * n * n) / 3
        e = (n * n) / 2
        i = n / 6
        return (a + e + i)
    return None
