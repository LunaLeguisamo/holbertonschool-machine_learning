#!/usr/bin/env python3
"""
Class that performs Bayesian optimization on a
noiseless 1D Gaussian process
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    f is the black-box function to be optimized
    X_init is a numpy.ndarray of shape (t, 1) representing the inputs
    already sampled with the black-box function
    Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
    of the black-box function for each input in X_init
    t is the number of initial samples
    bounds is a tuple of (min, max) representing the bounds of the space
    in which to look for the optimal point
    ac_samples is the number of samples that should be analyzed during
    acquisition
    l is the length parameter for the kernel
    sigma_f is the standard deviation given to the output of the black-box
    function
    xsi is the exploration-exploitation factor for acquisition
    minimize is a bool determining whether optimization should be performed
    for minimization (True) or maximization (False)
    Sets the following public instance attributes:
    f: the black-box function
    gp: an instance of the class GaussianProcess
    X_s: a numpy.ndarray of shape (ac_samples, 1) containing all acquisition
    sample points, evenly spaced between min and max
    xsi: the exploration-exploitation factor
    minimize: a bool for minimization versus maximization
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        f: black-box function
        X_init: numpy.ndarray of shape (t, 1) with initial inputs
        Y_init: numpy.ndarray of shape (t, 1) with initial outputs
        bounds: tuple (min, max), search space
        ac_samples: number of candidate acquisition points
        l: kernel length parameter
        sigma_f: GP output std
        xsi: exploration-exploitation factor
        minimize: True if minimizing, False if maximizing
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Method that calculates the next best sample location
        Uses the Expected Improvement acquisition function
        Returns: X_next, EI
        X_next is a numpy.ndarray of shape (1,) representing
        the next best sample point
        EI is a numpy.ndarray of shape (ac_samples,) containing
        the expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        # Best observed value so far
        Y_opt = np.min(self.gp.Y) if self.minimize else np.max(self.gp.Y)

        # Improvement
        if self.minimize:
            I = Y_opt - mu - self.xsi
        else:
            I = mu - Y_opt - self.xsi

        # Avoid division by zero in Z
        Z = np.zeros_like(I)
        mask = sigma > 0
        Z[mask] = I[mask] / sigma[mask]

        # Expected Improvement
        EI = I * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Next best sample
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Method that optimizes the black-box function:
        iterations is the maximum number of iterations to perform
        If the next proposed point is one that has already been sampled,
        optimization should be stopped early
        Returns: X_opt, Y_opt
        X_opt is a numpy.ndarray of shape (1,) representing the
        optimal point
        Y_opt is a numpy.ndarray of shape (1,) representing the
        optimal function value
        """
        for i in range(iterations):
            X_next, EI = self.acquisition()
            if np.any(np.all(self.gp.X == X_next, axis=1)):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            if self.minimize:
                idx_opt = np.argmin(self.gp.Y)
            else:
                idx_opt = np.argmax(self.gp.Y)
            X_opt = self.gp.X[idx_opt].reshape(1,)
            Y_opt = self.gp.Y[idx_opt].reshape(1,)

        return X_opt, Y_opt
