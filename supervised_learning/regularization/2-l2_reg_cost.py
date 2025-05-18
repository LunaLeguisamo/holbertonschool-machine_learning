#!/usr/bin/env python3
"""L2 Regularization Cost"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: Tensor, the base cost (without regularization).
        model: Keras model with layers that may include L2 regularization.

    Returns:
        Tensor with the total cost per layer (cost + layer's L2 losses).
    """
    total_costs = []

    for layer in model.layers:
        if layer.losses:  # If the layer has regularization losses
            # Sum of all regularization terms in this layer
            layer_l2 = tf.add_n(layer.losses)
            total = cost + layer_l2
            total_costs.append(total)

    return tf.convert_to_tensor(total_costs)
