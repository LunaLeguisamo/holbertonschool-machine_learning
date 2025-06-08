#!/usr/bin/env python3
"""
Build a dense block
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in DenseNet-B.

    Parameters:
    - X: input tensor
    - nb_filters: number of filters in the input tensor
    - growth_rate: growth rate (number of filters to add per layer)
    - layers: number of layers in the dense block

    Returns:
    - concatenated output of the dense block
    - updated number of filters
    """
    he_init = K.initializers.he_normal(seed=0)

    for i in range(layers):
        # BatchNorm + ReLU
        bn1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation('relu')(bn1)
        # 1x1 bottleneck convolution
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding='same',
            kernel_initializer=he_init,
            use_bias=False)(act1)

        # BatchNorm + ReLU
        bn2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        # 3x3 convolution
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer=he_init,
            use_bias=False)(act2)

        # Concatenate input with output of this layer
        X = K.layers.Concatenate()([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
