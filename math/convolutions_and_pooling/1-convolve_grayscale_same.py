#!/usr/bin/env python3
"""
Function that performs a same
convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w)
    - kernel: numpy.ndarray of shape (kh, kw)

    Returns:
    - numpy.ndarray containing the convolved images with same padding
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calcular el padding necesario
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # Aplicar padding a las imágenes
    padded = np.pad(images, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)

    # Preparar el resultado
    output = np.zeros((m, h, w))

    # Aplicar la convolución
    for i in range(h):
        for j in range(w):
            slice = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))

    return output
