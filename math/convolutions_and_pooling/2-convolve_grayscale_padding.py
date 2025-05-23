#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Applies a convolution operation to multiple grayscale images
    using a specific kernel and custom zero-padding.

    Parameters
    ----------
    images : numpy.ndarray of shape (m, h, w)
        - m: number of grayscale images.
        - h: height of each image.
        - w: width of each image.

    kernel : numpy.ndarray of shape (kh, kw)
        - kh: height of the kernel.
        - kw: width of the kernel.
        - This is the filter used to perform the convolution.

    padding : tuple of (ph, pw)
        - ph: padding applied to the height (top and bottom).
        - pw: padding applied to the width (left and right).
        - Padding is done with zeros.

    Returns
    -------
    output : numpy.ndarray of shape (m, h_out, w_out)
        - Contains the result of the convolution for each image.
        - h_out = h + 2*ph - kh + 1
        - w_out = w + 2*pw - kw + 1
    """

    # Retrieve shapes
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Apply zero-padding to the images
    padded = np.pad(images,
                    pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant',
                    constant_values=0)

    # Calculate the output dimensions
    h_output = h + 2 * ph - kh + 1
    w_output = w + 2 * pw - kw + 1

    # Initialize the output array
    output = np.zeros((m, h_output, w_output))

    # Perform the convolution
    for i in range(h_output):
        for j in range(w_output):
            slice = padded[:, i:i+kh, j:j+kw]       # shape: (m, kh, kw)
            # Sum across height and width
            output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))

    return output
