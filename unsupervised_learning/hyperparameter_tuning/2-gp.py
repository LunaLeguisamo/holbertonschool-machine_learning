#!/usr/bin/env python3
"""
Class hat represents a noiseless 1D Gaussian process
"""
import numpy as np


class GaussianProcess:
    """
    X_init is a numpy.ndarray of shape (t, 1) representing the
    inputs already sampled with the black-box function
    Y_init is a numpy.ndarray of shape (t, 1) representing the
    outputs of the black-box function for each input in X_init
    t is the number of initial samples
    l is the length parameter for the kernel
    sigma_f is the standard deviation given to the output of the
    black-box function
    Sets the public instance attributes X, Y, l, and sigma_f
    corresponding to the respective constructor inputs
    Sets the public instance attribute K, representing the current
    covariance kernel matrix for the Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        X_init: numpy.ndarray (t, 1) inputs already sampled
        Y_init: numpy.ndarray (t, 1) outputs of the black-box function
        l: length parameter
        sigma_f: standard deviation of output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        # Covariance matrix of the initial points
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Computes the covariance kernel matrix using RBF
        X1: numpy.ndarray (m, 1)
        X2: numpy.ndarray (n, 1)
        Returns: covariance matrix (m, n)
        """
        # Expand dimensions to compute pairwise squared distances
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

        return (self.sigma_f ** 2) * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        predicts the mean and standard deviation of points in
        a Gaussian process
        X_s is a numpy.ndarray of shape (s, 1) containing all
        of the points whose mean and standard deviation should
        be calculated s is the number of sample points
        Returns: mu, sigma
        mu is a numpy.ndarray of shape (s,) containing the mean
        for each point in X_s, respectively
        sigma is a numpy.ndarray of shape (s,) containing the
        variance for each point in X_s, respectively
        """
        # 1) Inversa de K (t x t)
        K_inv = np.linalg.inv(self.K)

        # 2) Kernels cruzados
        K_s = self.kernel(self.X, X_s)   # (t x s)
        K_ss = self.kernel(X_s, X_s)      # (s x s)

        # 3) Media (s,1) -> (s,)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)

        # 4) Covarianza (s x s) y varianza diagonal (s,)
        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov)

        return mu, sigma

    
    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process:
        X_new is a numpy.ndarray of shape (1,) that
        represents the new sample point
        Y_new is a numpy.ndarray of shape (1,) that
        represents the new sample function value
        Updates the public instance attributes X, Y, and K
        """
        X_new = X_new.reshape(-1, 1)
        Y_new = Y_new.reshape(-1, 1)
        self.X = np.concatenate((self.X, X_new), axis=0)
        self.Y = np.concatenate((self.Y, Y_new), axis=0)
        self.K = self.kernel(self.X, self.X)
