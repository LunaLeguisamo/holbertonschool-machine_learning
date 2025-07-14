#!/usr/bin/env python3
"""
A class Binomial that represents a binomial distribution
"""


class Binomial:
    """
    Represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution
        """
        if data is None:
            # Validaciones para n y p cuando no hay data
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Validaciones sobre data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calcular la media y varianza muestral
            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimar p y n
            p_est = 1 - (var / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est

            self.n = int(n_est)
            self.p = float(p_est)
