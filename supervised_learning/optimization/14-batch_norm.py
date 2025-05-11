#!/usr/bin/env python3
"""
Function  that creates a
normalization layer for a neural network
in tensorflow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be
    used on the output of the layer
    you should use the tf.keras.layers.Dense layer as the
    base layer with kernal initializer tf.keras.initializers.
    VarianceScaling(mode='fan_avg')
    your layer should incorporate two trainable parameters,
    gamma and beta, initialized as vectors of 1 and 0 respectively
    you should use an epsilon of 1e-7
    Returns: a tensor of the activated output for the layer
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n, kernel_initializer=initializer)(prev)
    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1, epsilon=1e-7)(dense)
    activated_output = activation(batch_norm)

    return activated_output
