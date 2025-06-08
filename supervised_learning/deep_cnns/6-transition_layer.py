#!/usr/bin/env python3
"""
Build a transition layer for DenseNet
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected Convolutional
    Networks following DenseNet-C architecture with compression.

    Args:
        X: Output tensor from the previous layer
        nb_filters: Integer representing the number of filters in X
        compression: Compression factor for the transition layer

    Returns:
        Tuple containing:
        - Output tensor of the transition layer
        - Integer representing the number of filters in the output
    """
    he_init = K.initializers.he_normal(seed=0)

    # Batch Normalization
    X = K.layers.BatchNormalization()(X)

    # ReLU activation (using ReLU layer instead of Activation('relu'))
    X = K.layers.ReLU()(X)

    # 1x1 convolution with compression (including bias terms)
    nb_filters = int(nb_filters * compression)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=he_init,
        use_bias=True
    )(X)

    # Average pooling with stride 2
    X = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        padding='same'
    )(X)

    return X, nb_filters
