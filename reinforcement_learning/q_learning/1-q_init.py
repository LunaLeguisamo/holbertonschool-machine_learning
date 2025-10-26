#!/usr/bin/env python3
"""
Initialize the Q-Table
"""
import numpy as np


def q_init(env):
    """
    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray
    of zeros
    """
    state_space = env.observation_space.n
    action_space = env.action_space.n

    q_table = np.zeros((state_space, action_space))
    return q_table
