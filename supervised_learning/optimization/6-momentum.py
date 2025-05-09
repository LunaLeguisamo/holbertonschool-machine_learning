#!/usr/bin/env python3
"""
Function hat sets up the gradient
descent with momentum optimization
algorithm in TensorFlow
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    alpha is the learning rate.
    beta1 is the momentum weight.
    Returns: optimizer
    """

    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
