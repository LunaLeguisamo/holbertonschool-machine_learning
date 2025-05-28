#!/usr/bin/env python3

"""
Function that performs forward propagation
over a convolutional layer of a neural network
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
    kh is the filter height
    kw is the filter width
    c_prev is the number of channels in the previous layer
    c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution activation is an activation function
    applied to the convolution padding is a string that is either same
    or valid, indicating the type of padding used stride is a tuple
    of (sh, sw) containing the strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    you may import numpy as np
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    if padding == 'valid':
        ph, pw = 0, 0

    padding = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant', constant_values=0)

    h_output = (h_prev + 2 * ph - kh // sh) + 1
    w_output = (w_prev + 2 * pw - kw // sw) + 1

    output = np.zeros((m, h_output, w_output, c_new))

    for i in range(m):
        for j in range(h_output):
            for k in range(w_output):
                for c in range(c_new):
                    start_h = j * sh
                    end_h = start_h + kh
                    start_w = k * sw
                    end_w = start_w + kw
                    slice = padding[i, start_h:end_h, start_w:end_w, :]
                    output[i, j, k, c] = np.sum(
                        slice * W[:, :, :, c]) + b[0, 0, 0, c]

    return activation(output)
