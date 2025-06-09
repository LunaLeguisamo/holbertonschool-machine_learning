#!/usr/bin/env python3
"""
Build a transition layer for DenseNet
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer for DenseNet with compression.
    
    Args:
        X: Tensor, input from previous layer
        nb_filters: int, number of filters before the transition
        compression: float, compression factor (0 < compression <= 1)

    Returns:
        Tuple: (output tensor, new number of filters)
    """
    he_init = K.initializers.he_normal(seed=0)
    
    # Batch Norm
    X = K.layers.BatchNormalization(axis=3, name=None)(X)
    
    # ReLU
    X = K.layers.Activation('relu')(X)
    
    # 1x1 Convolution (no bias needed after BatchNorm)
    nb_filters = int(nb_filters * compression)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=he_init,
        use_bias=False)(X)

    # 2x2 Average Pooling
    X = K.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(X)

    return X, nb_filters
