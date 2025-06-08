#!/usr/bin/env python3
"""
Build a transition layer as described in DenseNet-C.
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer.

    Parameters:
    - X: input tensor
    - nb_filters: number of filters in the input tensor
    - compression: compression factor for the transition layer

    Returns:
    - The output of the transition layer
    - The number of filters within the output
    """
    # He normal initializer with seed=0
    he_init = K.initializers.he_normal(seed=0)

    # Apply compression
    compressed_filters = int(nb_filters * compression)

    # Batch Normalization
    bn = K.layers.BatchNormalization()(X)
    # ReLU Activation
    relu = K.layers.Activation('relu')(bn)
    # 1x1 Convolution to reduce number of filters
    conv = K.layers.Conv2D(
        filters=compressed_filters,
        kernel_size=1,
        padding='same',
        kernel_initializer=he_init,
        use_bias=False
    )(relu)
    # 2x2 Average Pooling to reduce spatial dimensions
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2)(conv)

    return avg_pool, compressed_filters
