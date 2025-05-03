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
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
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

    if save_best and filepath:
        checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                 save_best_only=True)
    cllbk.append(checkpoint)

    return network.fit(
        data, labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=cllbk
    )
