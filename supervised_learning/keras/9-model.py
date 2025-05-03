#!/usr/bin/env python3
"""
This module defines utility functions to save and load
entire Keras models.

Functions:
- save_model: saves a Keras model (architecture, weights,
optimizer state) to a file
- load_model: loads a Keras model from a saved file
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file.

    Args:
        network (keras.Model): the Keras model
        to save filename (str): the path to the
        file to save the model to

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire Keras model from a file.

    Args:
        filename (str): the path to the file from
        which to load the model

    Returns:
        keras.Model: the loaded Keras model
    """
    return K.models.load_model(filename)
