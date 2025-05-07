#!/usr/bin/env python3
"""
Function that creates mini-batches from two matrices X and Y
in a synchronized way, to be used in mini-batch gradient descent.
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Splits the dataset (X, Y) into mini-batches of size `batch_size`.

    Parameters:
    -----------
    X : numpy.ndarray of shape (m, nx)
        Input data, where m is the number of samples and nx
        the number of features.
    Y : numpy.ndarray of shape (m, ny)
        Labels, where ny is the number of output classes or
        label dimension.
    batch_size : int
        The desired number of samples per mini-batch.

    Returns:
    --------
    mini_batches : list of tuples
        A list where each element is a tuple (X_batch, Y_batch).
        X_batch has shape
        (batch_size, nx) and Y_batch has shape (batch_size, ny),
        except possibly the last
        batch which may be smaller if m is not divisible by
        batch_size.

    Process:
    --------
    1. Shuffle X and Y in unison using `shuffle_data` to avoid bias
    in batch order.
    2. Partition the shuffled data into consecutive mini-batches of
    length `batch_size`.
    3. If the total number of samples m is not a multiple of
    `batch_size`,
       include a final mini-batch with the remaining samples.
    """
    # Number of training examples
    m = X.shape[0]

    # 1. Shuffle (X, Y) in unison to randomize batch composition
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    mini_batches = []
    # Compute the number of full mini-batches
    num_complete_batches = m // batch_size

    # 2. Create each full mini-batch
    for k in range(num_complete_batches):
        start = k * batch_size
        end = start + batch_size

        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    # 3. Handle the last batch (if any samples remain)
    if m % batch_size != 0:
        start = num_complete_batches * batch_size
        X_batch = X_shuffled[start:]
        Y_batch = Y_shuffled[start:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
