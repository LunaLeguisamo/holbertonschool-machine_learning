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
        return int(n * (n + 1) * (2 * n + 1) / 6)
    return None
