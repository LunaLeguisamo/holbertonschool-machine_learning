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
    concat = X
    for i in range(layers):
        # BatchNorm + ReLU
        bn = K.layers.BatchNormalization()(concat)
        act = K.layers.Activation('relu')(bn)
        # 1x1 bottleneck convolution
        bttleneck = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding='same',
            kernel_initializer=he_init,
            use_bias=False)(X)

        # BatchNorm + ReLU
        bn2 = K.layers.BatchNormalization()(X)
        act2 = K.layers.Activation('relu')(X)
        # 3x3 convolution
        conv = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer=he_init,
            use_bias=False)(X)

        # Concatenate input with output of this layer
        concat = K.layers.Concatenate()([concat, X])
        nb_filters += growth_rate
        
    return concat, nb_filters
