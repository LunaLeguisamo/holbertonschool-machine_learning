#!/usr/bin/env python3
"""
This script contains a function that calculates the integral of a polynomial
represented as a list of coefficients.

The poly_integral function takes a polynomial (list of coefficients) and a
constant C (integration constant, default 0),and returns a new list
representing the integrated polynomial.

The polynomial is represented as a list of coefficients, where the index of
each value in the list corresponds to the exponent of the variable x.

For example, the polynomial 3x^2 + 2x + 1 would be represented as [3, 2, 1].

The function integrates each term of the polynomial, incrementing the exponent
by 1 and dividing each coefficient by the new exponent. If the coefficient
is 0, it adds 0 at the corresponding position.

Returns:
    new_poly (list): List of coefficients of the integrated polynomial.
"""


def poly_integral(poly, C=0):
    """
    Parameters:
    poly (list): List of coefficients of the polynomial to integrate.
    C (int): Constant of integration. Defaults to 0.
    """
    # Check if poly is a valid list and if C is an integer
    if not isinstance(poly, list) or poly == []:
        return None

    if not isinstance(C, int):
        return None

    new_poly = []
    new_poly.append(C)

    for i in range(len(poly)):
        if poly[i] == 0:
            # If the coefficient is 0, append 0 to the result
            new_poly.append(0)
        else:
            # Perform the integration of each term
            if poly[i] % (i + 1) == 0:
                # If divisible, use integer division
                new_value = poly[i] // (i + 1)
            else:
                # If not divisible, use float division
                new_value = poly[i] / (i + 1)

            # Append the integrated coefficient to the result
            new_poly.append(new_value)
    return new_poly
