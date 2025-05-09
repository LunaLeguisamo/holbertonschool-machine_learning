#!/usr/bin/env python3
"""
Function that sets up the RMSProp
optimization algorithm in TensorFlow
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    alpha is the learning rate
    beta2 is the RMSProp weight (Discounting factor)
    epsilon is a small number to avoid division by zero
    Returns: optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        epsilon=epsilon,
        rho=beta2
        )
    return optimizer
