#!/usr/bin/env python3
"""
Function  that calculates a correlation matrix
"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Initialize the distribution with data.
        data: numpy.ndarray of shape (d, n) with the data set.
        TypeError: If data is not a 2D numpy array.
        ValueError: If n < 2.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Compute mean vector of shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data
        centered = data - self.mean

        # Compute covariance matrix manually: cov = (X · Xᵀ) / (n - 1)
        self.cov = (centered @ centered.T) / (n - 1)
        self.d = d

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        # Constantes
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        norm_const = 1 / (np.sqrt((2 * np.pi) ** self.d * det))

        # Exponente
        delta = x - self.mean  # (d, 1)
        exponent = -0.5 * (delta.T @ inv @ delta)

        # PDF
        pdf_value = float(norm_const * np.exp(exponent))

        return pdf_value
