#!/usr/bin/env python3

"""
Module that provides a function to calculate the summation
of squares of integers from 1 to n using a closed formula.
"""


def summation_i_squared(n):
    """
    Calculates the summation of squares from 1 to n.

    Uses the formula: sum(i^2) = n(n+1)(2n+1)/6

    Args:
        n (int): The stopping condition (must be a positive integer)

    Returns:
        int: The result of the summation if n is valid, otherwise None
    """
    if n > 0 and isinstance(n, int):
        a = (n * n * n) / 3
        e = (n * n) / 2
        i = n / 6
        return int(a + e + i)
    return None
