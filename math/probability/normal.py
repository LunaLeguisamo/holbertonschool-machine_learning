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
                variance = sum(squared_diffs) / len(data)
                self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        z-score
        """
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
        pi = 3.141592653589793
        e = 2.718281828459045
        x = float(x)

        num = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        den = self.stddev * (2 * pi) ** 0.5
        return num / den

    def _erf(self, z):
        """Aproximación de la función error"""
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        e = 2.718281828459045
        
        sign = 1
        if z < 0:
            sign = -1
            z = -z
        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (e ** (-z * z))

        return sign * y

    def cdf(self, x):
        """
        Cumulative Distribution Function
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self._erf(z))
