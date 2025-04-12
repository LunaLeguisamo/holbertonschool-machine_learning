#!/usr/bin/env python3

def poly_derivative(poly):
    if not isinstance(poly, list):
        return None
    new_poly = []
    for pol in poly:
        new_poly.append(pol)
        
    for i in range(len(poly)):
        new_poly.append(i[len(poly)-1])
