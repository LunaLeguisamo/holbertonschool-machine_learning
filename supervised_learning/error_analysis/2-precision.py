#!/usr/bin/env python3

"""
Function  that creates a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    """
    classes = confusion.shape[0]
    precision = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        precision[i] = TP / (TP + FP)

    return precision
