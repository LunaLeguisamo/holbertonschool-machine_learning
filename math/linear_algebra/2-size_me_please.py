#!/usr/bin/env python3

def matrix_shape(matrix):
    if not isinstance(matrix[0], list):
        return [len(matrix)]
    
    return [len(matrix)] + matrix_shape(matrix[0])
