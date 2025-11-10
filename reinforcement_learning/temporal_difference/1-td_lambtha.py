#!/usr/bin/env python3
"""
1-td_lambtha.py
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Perform the TD(λ) algorithm.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to
        take
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    # Initialize eligibility traces
    E = np.zeros_like(V)

    for episode in range(episodes):
        # Reset environment and eligibility traces for new episode
        state = env.reset()[0]
        E.fill(0)  # Reset eligibility traces

        for step in range(max_steps):
            # Choose action according to policy
            action = policy(state)

            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Calculate TD error
            td_error = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace for current state
            E[state] += 1

            # Update all states according to their eligibility traces
            V += alpha * td_error * E

            # Decay eligibility traces
            E *= gamma * lambtha

            # Update current state
            state = next_state

            # Check if episode ended
            if terminated or truncated:
                break

    return V
