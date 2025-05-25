#!/usr/bin/env python3
"""
Function that performs a convolution
on grayscale images
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)): 
    """
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernels.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == "valid":
        ph, pw = 0, 0
    elif isinstance(padding, tuple):
        ph, pw = padding

    padding = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant', constant_values=0)

    h_output = (h + 2 * ph - kh) // sh + 1
    w_output = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_output, w_output))

    for i in range(h_output):
        for j in range(w_output):
            slice = padding[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            output[:, i, j] = np.sum(slice * kernels[i][j], axis=(1, 2, 3))

    return output