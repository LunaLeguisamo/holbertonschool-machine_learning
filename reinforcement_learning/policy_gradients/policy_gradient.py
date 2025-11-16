#!/usr/bin/env python3
"""
Funcion that computes the policy with a weight of a matrix.
"""
import numpy as np


def policy(matrix, weight):
    """Calculates the action to take using a policy gradient

    Args:
        matrix (numpy.ndarray): 2D array representing the state
        weight (numpy.ndarray): 2D array representing the weights

    Returns:
        numpy.ndarray: 1D array of action probabilities
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))  # for numerical stability
    return exp / exp.sum(axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Computes the action and gradient using policy gradient

    Args:
        state (numpy.ndarray): 1D array representing the current state
        weight (numpy.ndarray): 2D array representing the weights

    Returns:
        tuple: action (int), gradient (numpy.ndarray)
    """
    state = state.reshape(1, -1)  # Reshape state to 2D array
    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Create one-hot encoding for the action
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1

    # Compute the gradient
    grad = np.dot(state.T, (one_hot - probs))

    return action, grad
