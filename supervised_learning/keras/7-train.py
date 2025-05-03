#!/usr/bin/env python3
"""
This module defines a function to train a Keras model using:

- Supplied training data and labels
- Configurable batch size, number of epochs, verbosity, and shuffling
- Optional validation data for monitoring performance
- Optional early stopping to prevent overfitting
- Optional learning rate decay using inverse time decay
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a Keras model with optional early stopping and learning rate decay.

    Parameters:
    - network: compiled Keras model to train
    - data: numpy.ndarray of input data
    - labels: numpy.ndarray of correct labels (one-hot encoded)
    - batch_size: size of batches for mini-batch gradient descent
    - epochs: number of passes through data for training
    - validation_data: tuple (data, labels) for validation, if any
    - early_stopping: boolean to indicate if early stopping should be used
    - patience: number of epochs with no improvement after which training
    stops
    - learning_rate_decay: boolean to indicate if learning rate decay
    should be used
    - alpha: initial learning rate
    - decay_rate: decay rate for inverse time decay
    - verbose: boolean to control verbosity of training output
    - shuffle: boolean to indicate if training data should be
    shuffled each epoch

    Returns:
    - History object generated after training the model
    """

    cllbk = []

    # Early stopping callback
    if validation_data and early_stopping:
        early_stp = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        cllbk.append(early_stp)

    # Learning rate scheduler callback
    if validation_data and learning_rate_decay:
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_scheduler = K.callbacks.LearningRateScheduler(schedule, verbose=1)
        cllbk.append(lr_scheduler)

    # Train the model with callbacks
    return network.fit(
        data, labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=cllbk
    )
