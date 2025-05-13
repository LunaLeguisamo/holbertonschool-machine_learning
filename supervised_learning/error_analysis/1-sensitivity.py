#!/usr/bin/env python3

"""
Function  that creates a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    """
    classes = confusion.shape[0]
    sensitivities = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i, i]
        FN = np.sum(confusion[i, :]) - TP
        sensitivities[i] = TP / (TP + FN)

    return sensitivities
