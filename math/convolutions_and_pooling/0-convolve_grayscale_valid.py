#!/usr/bin/env python3
"""
Function that performs a valid convolution on grayscale images:
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w)
    - kernel: numpy.ndarray of shape (kh, kw)

    Returns:
    - numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    new_h = h - kh + 1
    new_w = w - kw + 1

    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            # Extraer la porción de cada imagen
            # que se corresponde con el kernel
            slice = images[:, i:i+kh, j:j+kw]
            # Realizar la multiplicación elemento a
            # elemento y sumar para obtener un escalar por imagen
            output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))

    return output
