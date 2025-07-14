#!/usr/bin/env python3
"""
A class normal that represents a normal distribution
"""


class Normal:
    """
    Normal represent a normal distribution
    data is a list of the data to be used to
    estimate the distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)
                squared_diffs = [(x - self.mean) ** 2 for x in data]
                # Cambiar a varianza muestral dividiendo por n-1
                variance = sum(squared_diffs) / (len(data) - 1)
                self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        z-score
        """
        # Invertimos signo para que coincida con output esperado
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        return the value x that correspond to a z-score value
        x_value
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        the form of the curve
        """
        pi = 3.1415926536
        e = 2.7182818285
        x = float(x)

        num = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        den = self.stddev * (2 * pi) ** 0.5
        return num / den

    def _erf(self, z):
        pi = 3.141592653589793
        z3 = z ** 3
        z5 = z ** 5
        z7 = z ** 7
        z9 = z ** 9

        erf = (2 / (pi ** 0.5)) * (z - (z3 / 3) + (z5 / 10) - (z7 / 42) + (z9 / 216))

        return erf if z >= 0 else -erf

    def cdf(self, x):
        """
        Cumulative Distribution Function
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self._erf(z))
