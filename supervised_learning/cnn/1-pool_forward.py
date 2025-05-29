#!/usr/bin/env python3
"""
Function that performs forward propagation
over a pooling layer of a neural network
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of
    the kernel for the pooling
    kh is the kernel height
    kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height
    sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    you may import numpy as np
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_output = ((h_prev - kh) // sh) + 1
    w_output = ((w_prev - kw) // sw) + 1

    output = np.zeros((m, h_output, w_output, c_prev))

    for i in range(m):
        for j in range(h_output):
            for k in range(w_output):
                for c in range(c_prev):
                    start_h = j * sh
                    end_h = start_h + kh
                    start_w = k * sw
                    end_w = start_w + kw
                    slicing = A_prev[i, start_h:end_h, start_w:end_w, c]
                    if mode == 'max':
                        output[i, j, k, c] = np.max(slicing)
                    elif mode == 'avg':
                        output[i, j, k, c] = np.mean(slicing)

    return output
