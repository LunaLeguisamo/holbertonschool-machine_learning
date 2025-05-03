#!/usr/bin/env python3
"""
This module defines a function that trains a Keras model using:

- Provided training data and labels
- Configurable batch size, number of epochs, verbosity, and shuffling
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a Keras model on given data.

    Parameters:
    - network: keras.Model
        The compiled Keras model to train.
    - data: numpy.ndarray or tensor
        Input data of shape (m, nx), where m is the number
        of examples and nx is the number of features.
    - labels: numpy.ndarray or tensor
        One-hot encoded labels of shape (m, classes), where
        classes is the number of output classes.
    - batch_size: int
        The number of samples per gradient update.
    - epochs: int
        The number of passes through the entire dataset.
    - verbose: bool (default=True)
        Whether to print progress messages during training.
    - shuffle: bool (default=False)
        Whether to shuffle the training data before each epoch.
    - validation_data: (default=None)

    Returns:
    - A History object generated after training, which contains information
      such as loss and accuracy metrics over the epochs.
    """

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
