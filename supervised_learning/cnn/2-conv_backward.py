#!/usr/bin/env python3
"""
Function hat performs back propagation over
a convolutional layer of a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the unactivated output of
    the convolutional layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
    kh is the filter height
    kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    padding is a string that is either same or valid, indicating the type
    of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    you may import numpy as np
    Returns: the partial derivatives with respect to the previous layer (dA_prev),
    the kernels (dW), and the biases (db), respectively
    """
    
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Calcular padding si es necesario
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = padding
        
    # Padding para A_prev y su gradiente
    A_prev_pad = np.pad(A_prev,
                        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    #dA_prev_pad = np.pad(dA_prev,
    #                     pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
    #\                     mode='constant')
    
    # Inicializar gradientes
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    
    # Loop para cada ejemplo del batch
    for h in range(h_new):
        for w in range(w_new):
            for c in range(c_new):
                # Encontrar corners de la ventana
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                # Calcular gradientes
                dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += (W[:, :, :, c] * dZ[:, h, w, c, np.newaxis, np.newaxis, np.newaxis])
                dW[:, :, :, c] += np.sum(A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] * 
                                            dZ[:, h, w, c, np.newaxis, np.newaxis, np.newaxis], axis=0)

    # Recortar padding
    if padding == 'same':
        dA_prev_pad = dA_prev_pad[:, ph:-ph, pw:-pw, :]

    return dA_prev_pad, dW, db