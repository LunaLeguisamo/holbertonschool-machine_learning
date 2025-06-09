#!/usr/bin/env python3
"""
Build a dense block
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in DenseNet (with bottleneck).
    
    Args:
        X: Tensor output from previous layer
        nb_filters: int, current number of filters
        growth_rate: int, growth rate
        layers: int, number of layers in the block

    Returns:
        Tuple: (output tensor of the dense block, updated number of filters)
    """
    he_init = K.initializers.he_normal(seed=0)

    for i in range(layers):
        # BatchNorm + ReLU
        X1 = K.layers.BatchNormalization(axis=3)(X)
        X1 = K.layers.Activation('relu')(X1)

        # 1x1 bottleneck convolution (DenseNet-B)
        X1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding='same',
            kernel_initializer=he_init,
            use_bias=False)(X1)

        # BatchNorm + ReLU
        X1 = K.layers.BatchNormalization(axis=3)(X1)
        X1 = K.layers.Activation('relu')(X1)

        # 3x3 convolution
        X1 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer=he_init,
            use_bias=False)(X1)

        # Concatenate with previous layers
        X = K.layers.Concatenate(axis=3)([X, X1])
        nb_filters += growth_rate

    return X, nb_filters
