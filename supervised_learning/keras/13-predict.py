#!/usr/bin/env python3
"""
This module defines a function to make predictions using a neural network.

Function:
- predict: performs a prediction on the given data using the trained model.
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using the provided neural network.

    Args:
        network (keras.Model): the model to make predictions with
        data (numpy.ndarray): the input data to make predictions on
        verbose (bool): whether to print output during the prediction process

    Returns:
        numpy.ndarray: the predictions made by the network
    """
    return network.predict(data, verbose=verbose)
