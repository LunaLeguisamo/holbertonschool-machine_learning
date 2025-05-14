#!/usr/bin/env python3

"""
Function  that creates a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    """
    classes = confusion.shape[0]
    specificity = np.zeros(classes)

    total = np.sum(confusion)

    for i in range(classes):
        TP = confusion[i, i]
        FN = np.sum(confusion[i, :]) - TP
        FP = np.sum(confusion[:, i]) - TP
        TN = total - TP - FP - FN
        specificity[i] = TN / (TN + FP)

    return specificity
