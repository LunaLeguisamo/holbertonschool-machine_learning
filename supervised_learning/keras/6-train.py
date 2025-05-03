#!/usr/bin/env python3
"""
This module defines a function that trains a Keras model using:

- Provided training data and labels
- Configurable batch size, number of epochs, verbosity, and shuffling
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    comments
    """
    if validation_data and early_stopping:
        early_stp = [K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)]
    else:
        early_stp = None
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, verbose=verbose,
                       shuffle=shuffle, callbacks=early_stp)
