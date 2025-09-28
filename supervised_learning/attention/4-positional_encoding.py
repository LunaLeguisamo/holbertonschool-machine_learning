#!/usr/bin/env python3
"""
Function that calculates the positional
encoding for a transformer
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    max_seq_len is an integer representing the maximum sequence length
    dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the
    positional encoding vectors
    """
    # Crear matriz de posiciones
    position = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)

    # Usamos i // 2 en lugar de i
    i = np.arange(dm)  # [0, 1, 2, 3, ..., dm-1]
    div_term = np.power(10000, (2 * (i // 2)) / dm)  # (dm,)

    # Calcular ángulos
    angle_rads = position / div_term  # (max_seq_len, dm)

    # Aplicar seno a índices pares, coseno a impares
    PE = np.zeros((max_seq_len, dm))
    PE[:, 0::2] = np.sin(angle_rads[:, 0::2])  # índices pares
    PE[:, 1::2] = np.cos(angle_rads[:, 1::2])  # índices impares

    return PE
