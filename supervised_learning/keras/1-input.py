#!/usr/bin/env python3
"""
This module defines a function that builds a fully connected neural
network
using the Keras Functional API. It includes support for:
- Multiple hidden layers
- Custom activation functions
- L2 regularization
- Dropout regularization
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using the Keras Functional API.

    Parameters:
    - nx (int): Number of input features.
    - layers (list): List of integers representing the number of nodes
    in each layer.
    - activations (list): List of activation functions for each layer.
    - lambtha (float): L2 regularization parameter (to reduce overfitting).
    - keep_prob (float): Probability of keeping a node active during dropout.

    Returns:
    - model (keras.Model): Compiled Keras model.
    """

    # Define the input layer with the shape of the input features
    inputs = K.Input(shape=(nx,))

    # First dense layer (without dropout before it)
    out = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    )(inputs)

    # Add the remaining layers with dropout before each
    for i in range(1, len(layers)):
        # Apply Dropout with dropout rate = 1 - keep_prob
        out = K.layers.Dropout(1 - keep_prob)(out)

        # Add the next Dense layer
        out = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(out)

    # Create the Keras model object
    model = K.Model(inputs=inputs, outputs=out)
    return model
