#!/usr/bin/env python3
"""
"""


def poly_integral(poly, C=0):
    if not isinstance(poly, list) or poly == []:
        return None

    if not isinstance(C, int):
        return None

    new_poly = []

    for i in range(len(poly)):
        if i == 0:
            new_poly.append(C)
        elif poly[i] == 0:
            new_poly.append(0)
        if poly[i] % (i + 1) == 0:
            new_value = poly[i] // (i + 1)
        else:
            new_value = poly[i] / (i + 1)
        new_poly.append(new_value)
    return new_poly
