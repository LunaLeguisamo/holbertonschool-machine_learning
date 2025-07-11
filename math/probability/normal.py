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
