#!/usr/bin/env python3

"""

"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    """
    m, h, w = images.shape

    kh, kw = kernel.shape
    new_h = h - kh + 1
    new_w = w - kw + 1

    output = np.zeros((m, new_h, new_w))

    for i in range(kw):
        for j in range(kh):
            slice = images[:, i:i+kw, j:j+kh]
            output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))

    return output
