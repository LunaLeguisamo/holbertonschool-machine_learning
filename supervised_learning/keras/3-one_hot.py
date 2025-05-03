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


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix using
    tensorflow.keras.
    labels: numpy array or list
        A vector of labels to convert to one-hot encoding.
    classes: int, optional
        The number of classes. If None, it will be inferred
        from the maximum label in `labels`.
    Returns:
    tensor
        The one-hot encoded matrix as a tensor.
    """

    # If classes is not provided, we infer it from the max label + 1
    if classes is None:
        classes = K.backend.max(labels).numpy() + 1

    # Convert the labels to a one-hot encoded tensor
    one_hot_matrix = K.backend.one_hot(labels, classes)
    return one_hot_matrix.numpy()
