#!/usr/bin/env python3
"""
Create a class Poission
"""
import math as math


class Poisson:
    """
    A class represents a poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for
        a given number of “successes”
        """
        e = 2.7182818285
        self.k = int(k)
        if k < 0:
            return 0
        return e ** -self.lambtha * self.lambtha ** k / math.factorial(k)
