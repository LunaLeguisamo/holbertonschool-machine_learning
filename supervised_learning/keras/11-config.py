#!/usr/bin/env python3
"""
This module defines functions to save and load a Keras model's configuration.

Functions:
- save_config: saves the model's architecture in JSON format
- load_config: loads a model from a JSON configuration file
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.

    Args:
        network (keras.Model): the model whose configuration should be saved
        filename (str): the file path to save the configuration to

    Returns:
        None
    """
    config_json = network.to_json()
    with open(filename, 'w') as f:
        f.write(config_json)


def load_config(filename):
    """
    Loads a model with a specific configuration from JSON format.

    Args:
        filename (str): the file path containing the model’s configuration

    Returns:
        keras.Model: the model created from the configuration
    """
    with open(filename, 'r') as f:
        config_json = f.read()
    return K.models.model_from_json(config_json)
