#!/usr/bin/env python3
"""
Load and initialize the environment
for Q-learning.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False, render_mode="ansi"):
    """Load the Frozen Lake environment from OpenAI Gym.

    Args:
        desc (list of list of str, optional): Custom
        description of the map.
        map_name (str, optional): Predefined map name.
        is_slippery (bool): Whether the ice is slippery.

    Returns:
        gym.Env: The initialized Frozen Lake environment.
    """
    env = gym.make(
        'FrozenLake-v1', desc=desc, map_name=map_name,
        is_slippery=is_slippery, render_mode="ansi"
        )
    return env
