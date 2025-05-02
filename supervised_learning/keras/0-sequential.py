#!/usr/bin/env python3
"""
This module defines a function to build a sequential Keras model
with L2 regularization and dropout applied to each layer.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a Keras Sequential model with the given parameters.

    Args:
        nx (int): Number of input features.
        layers (list): List with the number of nodes for each layer.
        activations (list): List of activation functions for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability of keeping a node during dropout.

    Returns:
        keras.models.Sequential: The constructed Keras model.
    """
    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            ))
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
