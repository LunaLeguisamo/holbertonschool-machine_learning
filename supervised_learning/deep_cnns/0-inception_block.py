#!/usr/bin/env python3
"""
This module contains the implementation of an Inception
block as described in "Going Deeper with Convolutions."

The Inception block allows you to extract features at
different scales by combining 1x1, 3x3, 5x5 convolutions,
and parallel pooling.
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an Inception block.

    Parameters:
    - A_prev: output of the previous block (input tensor)
    - filters: tuple or list with 6 values:
    F1 -> filters for the 1x1 convolution
    F3R -> filters for the 1x1 convolution prior to 3x3
    F3 -> filters for the 3x3 convolution
    F5R -> filters for the 1x1 convolution prior to 5x5
    F5 -> filters for the 5x5 convolution
    FPP -> filters for the 1x1 convolution after max pooling

    Returns:
    - Tensor resulting from concatenating the outputs of all branches.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Rama 1: Convolución 1x1
    conv_1x1 = K.layers.Conv2D(
        filters=F1, kernel_size=(1, 1),
        padding='same', activation='relu'
    )(A_prev)

    # Rama 2: 1x1 seguido de 3x3
    conv_3x3_reduce = K.layers.Conv2D(
        filters=F3R, kernel_size=(1, 1),
        padding='same', activation='relu'
    )(A_prev)

    conv_3x3 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        padding='same', activation='relu'
    )(conv_3x3_reduce)

    # Rama 3: 1x1 seguido de 5x5
    conv_5x5_reduce = K.layers.Conv2D(
        filters=F5R, kernel_size=(1, 1),
        padding='same', activation='relu'
    )(A_prev)

    conv_5x5 = K.layers.Conv2D(
        filters=F5, kernel_size=(5, 5),
        padding='same', activation='relu'
    )(conv_5x5_reduce)

    # Rama 4: Max Pooling seguido de 1x1
    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1),
        padding='same'
    )(A_prev)

    conv_pool_proj = K.layers.Conv2D(
        filters=FPP, kernel_size=(1, 1),
        padding='same', activation='relu'
    )(max_pool)

    # Concatenamos las 4 ramas en la dimensión de canales (axis=-1)
    output = K.layers.concatenate(
        [conv_1x1, conv_3x3, conv_5x5, conv_pool_proj], axis=-1)

    return output
