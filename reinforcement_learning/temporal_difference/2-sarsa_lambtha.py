#!/usr/bin/env python3
"""
2-sarsa_lambtha.py
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform SARSA(λ) algorithm.
    """

    for episode in range(episodes):
        # Reset environment
        state = env.reset()[0]

        # Initialize eligibility traces for this episode
        E = np.zeros_like(Q)

        # Choose initial action using epsilon-greedy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        for step in range(max_steps):
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action using epsilon-greedy
            if np.random.uniform() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            # Calculate TD error
            td_error = reward + gamma * Q[next_state, next_action] -\
                Q[state, action]

            # Update eligibility trace for current state-action pair
            E[state, action] += 1

            # Update Q-values using eligibility traces
            Q += alpha * td_error * E

            # Decay eligibility traces
            E *= gamma * lambtha

            # Move to next state and action
            state = next_state
            action = next_action

            if terminated or truncated:
                break

        # Update epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
