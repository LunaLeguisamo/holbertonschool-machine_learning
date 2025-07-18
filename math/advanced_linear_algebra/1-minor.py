#!/usr/bin/env python3
"""
function that calculates the minor matrix of a matrix
"""


def minor(matrix):
    """
    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message matrix
    must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix
    Returns: the minor matrix of matrix
    """
    minor = []
    minor0 = []
    minor1 = []
    
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise ValueError("matrix must be a non-empty square matrix")
    
    size = len(matrix)

    if size == 1:
        minor.append([1])
        return minor

    if size == 2:
        minor0.append(matrix[1][1])
        minor0.append(matrix[1][0])
        minor1.append(matrix[0][1])
        minor1.append(matrix[0][0])
        minor.append(minor0)
        minor.append(minor1)
        return minor
    
    for i, elem in enumerate(matrix[0]):
        minor_ = [row[:i] + row[i+1:] for row in matrix[1:]]
        minor0.append(minor_[0][0] * minor_[1][1] - minor_[0][1] * minor_[1][0])
        minor.append(minor0)
        
    return minor
