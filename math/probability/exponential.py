#!/usr/bin/env python3
"""
Create a class Exponential
"""


class Exponential:
    """
    Class exponential
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Exponential represents an
        exponential distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """
        Compute the probability density function
        """
        e = 2.7182818285
        x = float(x)
        lambtha = self.lambtha
        if x < 0:
            return 0
        else:
            pdf = lambtha * (e ** -(lambtha * x))
            return pdf
