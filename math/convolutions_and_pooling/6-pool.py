#!/usr/bin/env python3
"""
Function that performs a convolution
on grayscale images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing
    the kernel shape for the pooling
    kh is the height of the kernel
    kw is the width of the kernel
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    mode indicates the type of pooling
    max indicates max pooling
    avg indicates average pooling
    You are only allowed to use two for loops;
    any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_output = (h - kh) // sh + 1
    w_output = (w - kw) // sw + 1

    output = np.zeros((m, h_output, w_output, c))

    for i in range(h_output):
        for j in range(w_output):
            slice = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            if mode == 'max':
                output[:, i, j, :] = np.max(slice, axis=(1, 2))
            if mode == 'avg':
                output[:, i, j, :] = np.mean(slice, axis=(1, 2))

    return output
