#!/usr/bin/env python3
"""
Function that creates a neural network
layer in tensorFlow that includes L2 regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a layer with L2 regularization"""
    return tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )(prev)
