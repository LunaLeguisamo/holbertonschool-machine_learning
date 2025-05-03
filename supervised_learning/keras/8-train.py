#!/usr/bin/env python3
"""
This module defines a function to train a Keras model using:

- Supplied training data and labels
- Configurable batch size, number of epochs, verbosity, and shuffling
- Optional validation data for monitoring performance
- Optional early stopping to prevent overfitting
- Optional learning rate decay using inverse time decay
- Optional checkpointing to save the best model based on validation loss
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    Trains a Keras model with configurable training options.

    Args:
        network (keras.Model): the model to train
        data (np.ndarray): input data for training
        labels (np.ndarray): correct labels for the input data
        batch_size (int): size of the batch used for training
        epochs (int): number of passes through the full dataset
        validation_data (tuple): data for validation (inputs, labels)
        early_stopping (bool): whether to stop early if val_loss
        stops improving patience (int): number of epochs with no
        improvement to wait before stopping
        learning_rate_decay (bool): whether to apply learning rate decay
        alpha (float): initial learning rate
        decay_rate (float): rate of decay for learning rate
        save_best (bool): whether to save the model with lowest validation
        loss
        filepath (str): path to save the best model
        (required if save_best is True)
        verbose (bool): whether to display training output
        shuffle (bool): whether to shuffle the training data every epoch

    Returns:
        History object generated after training the model
    """
    cllbk = []

    # Add EarlyStopping callback if required
    if validation_data and early_stopping:
        early_stp = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        cllbk.append(early_stp)

    # Add LearningRateScheduler callback if required
    if validation_data and learning_rate_decay:
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_scheduler = K.callbacks.LearningRateScheduler(schedule, verbose=1)
        cllbk.append(lr_scheduler)

    # Add ModelCheckpoint callback if saving the best model is required
    if save_best and filepath:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_best_only=True,
            monitor='val_loss'
        )
        cllbk.append(checkpoint)

    # Train the model
    return network.fit(
        data, labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=cllbk
    )
