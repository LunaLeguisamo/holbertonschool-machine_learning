#!/usr/bin/env python3
"""

"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh = stride[0]
    sw = stride[1]
    
    if padding == 'same':
        ph = kh // sh
        pw = kw // sw
        
        padding = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)))
        output = np.zeros((m, ph, pw))
        
        for i in range(h):
            for j in range(w):
                slice = padding[:, i:i + kh, j:j + kw]
                output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))
                
    elif padding == 'valid':
        new_h = (h - kh / sh) + 1
        new_w = (w - kw / sw) + 1
        
        output = np.zeros((m, new_h, new_w))
        
        for i in range(h):
            for j in range(w):
                slice = images[:, i:i + kh, j:j + kw]
                output[:, i, j] = np.sum(slice * kernel, axis=(1, 2)) 
                
    elif isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1] 
        h_output = (h + 2 * ph - kh / sh) + 1
        w_output = (w + 2 * pw - kw / sw) + 1
        
        padding = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)), mode='constant', constant_values=0)
        output = np.zeros((m, h_output, w_output))
        
        for i in range(h_output):
            for j in range(w_output):
                slice = padding[:, i:i + kh, j:j + kw]
                output[:, i, j] = np.sum(slice * kernel, axis=(1, 2))
            
    return output