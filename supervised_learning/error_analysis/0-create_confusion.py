#!/usr/bin/env python3

"""
Function  that creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    """
    true_classes = labels.argmax(axis=1)
    predicted_classes = logits.argmax(axis=1)

    num_classes = labels.shape[1]
    confusion = np.zeros((num_classes, num_classes))

    for t, p in zip(true_classes, predicted_classes):
        confusion[t][p] += 1

    return confusion
