#!/usr/bin/env python3
"""
This module defines a function that sets up the Adam optimizer
for a Keras model. It compiles the model using:

- Adam optimization with custom hyperparameters
- Categorical crossentropy loss
(for classification problems with one-hot labels)
- Accuracy as evaluation metric
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures and compiles a Keras model with the Adam optimizer.

    Parameters:
    - network (keras.Model): The Keras model to compile.
    - alpha (float): The learning rate for the Adam optimizer.
    - beta1 (float): The exponential decay rate for the 1st moment estimates.
    - beta2 (float): The exponential decay rate for the 2nd moment estimates.

    Returns:
    - None: The model is compiled in-place.
    """
    adam = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return None
