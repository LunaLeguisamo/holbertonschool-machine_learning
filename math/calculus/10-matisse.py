#!/usr/bin/env python3

"""
Module for computing the derivative of a polynomial.

The polynomial is represented as a list of coefficients.
Index represents the power of x.
Example: [5, 3, 0, 1] → 5 + 3x + 0x² + 1x³
The derivative is: [3, 0, 3]
"""


def poly_derivative(poly):
    """
    Computes the derivative of a polynomial.

    Args:
        poly (list): List of coefficients (ints or floats),
                     where index represents the power of x.

    Returns:
        list: Coefficients of the derivative polynomial,
              or None if input is not a list.
    """

    if not isinstance(poly, list):
        return None

    if len(poly) < 2:
        return [0]

    new_poly = []

    for i in range(1, len(poly)):
        new_poly.insert(i - 1, poly[i] * i)
    return new_poly
