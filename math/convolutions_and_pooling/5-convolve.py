#!/usr/bin/env python3
"""
Function that performs a convolution
on grayscale images
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernels is a numpy.ndarray with shape (kh, kw, c, nc)
    containing the kernels for the convolution
    kh is the height of a kernel
    kw is the width of a kernel
    nc is the number of kernels
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    You are only allowed to use three for loops; any other
    loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, kc = kernels.shape
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

    output = np.zeros((m, h_output, w_output, kc))

    for i in range(h_output):
        for j in range(w_output):
            for k in range(kc):
                slice = padding[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                kernel = kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(slice * kernel, axis=(1, 2, 3))

    return output
