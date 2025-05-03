#!/usr/bin/env python3
"""
This module defines a function to test a Keras neural network.

Function:
- test_model: evaluates the model with test data and returns loss and accuracy.
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network using provided data and labels.

    Args:
        network (keras.Model): the model to evaluate
        data (numpy.ndarray): the input test data
        labels (numpy.ndarray): the true one-hot encoded labels
        verbose (bool): whether to print evaluation output

    Returns:
        tuple: (loss, accuracy) on the test data
    """
    return network.evaluate(data, labels, verbose=verbose)
