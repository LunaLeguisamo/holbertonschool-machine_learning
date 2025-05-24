#!/usr/bin/env python3
"""
Function that performs a convolution
on grayscale images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing
    the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
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
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        ph = h - kh // 2 - 1
        pw = w - kw // 2 - 1

        padding = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)))

        h_output = (h + 2 * ph - kh) // sh + 1
        w_output = (w + 2 * pw - kw) // sw + 1

        output = np.zeros((m, h_output, w_output))

        for i in range(h_output):
            for j in range(w_output):
                slice = padding[:, i:i + kh, j:j + kw]
                output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))

    elif padding == 'valid':
        new_h = (h - kh) // sh + 1
        new_w = (w - kw) // sw + 1

        output = np.zeros((m, new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                slice = images[:, i:i + kh, j:j + kw]
                output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))

    elif isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
        h_output = (h + 2 * ph - kh) // sh + 1
        w_output = (w + 2 * pw - kw) // sw + 1

        padding = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                         mode='constant', constant_values=0)
        output = np.zeros((m, h_output, w_output))

        for i in range(h_output):
            for j in range(w_output):
                slice = padding[:, i:i + kh, j:j + kw]
                output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))

    return output
