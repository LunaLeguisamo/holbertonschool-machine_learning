#!/usr/bin/env python3
"""
Function that performs a convolution on
grayscale images with custom padding
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = padding[0]
    pw = padding[1]

    padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant', constant_values=0)

    h_output = h + 2*ph - kh + 1
    w_output = w + 2*pw - kw + 1
    output = np.zeros((m, h_output, w_output))

    for i in range(h):
        for j in range(w):
            slice = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))

    return output
