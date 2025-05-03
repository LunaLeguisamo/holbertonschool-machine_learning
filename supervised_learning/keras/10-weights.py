#!/usr/bin/env python3
"""
This module defines functions to save and load only the weights
of a Keras model.

Functions:
- save_weights: saves only the weights of a Keras model to a file
- load_weights: loads weights into a Keras model from a file
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves only the weights of a Keras model.

    Args:
        network (keras.Model): the model whose weights should be saved
        filename (str): the file path to save the weights to
        save_format (str): the format for saving the weights ('keras' or 'h5')

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads weights into a Keras model.

    Args:
        network (keras.Model): the model to load the weights into
        filename (str): the file path from which to load the weights

    Returns:
        None
    """
    network.load_weights(filename)
