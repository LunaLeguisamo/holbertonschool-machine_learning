#!/usr/bin/env python3

"""
Function  that creates a confusion matrix
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    """
    classes = confusion.shape[0]
    f1 = np.zeros(classes)

    recall = 0
    prec = 0
    recall = sensitivity(confusion)
    prec = precision(confusion)
    f1 = 2 * ((prec * recall) / (prec + recall))

    return f1
