#!/usr/bin/env python3
"""
2-sarsa_lambtha.py
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Perform SARSA(λ) algorithm.

    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: updated Q table
    """
    # Save initial epsilon for decay calculation
    initial_epsilon = epsilon

    for episode in range(episodes):
        # Initialize eligibility traces
        E = np.zeros_like(Q)
        # Reset environment and get initial state
        state = env.reset()[0]

        # Choose initial action using epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.argmax(Q[state])

        for step in range(max_steps):
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action using epsilon-greedy policy
            if np.random.uniform() < epsilon:
                next_action = np.random.randint(Q.shape[1])
            else:
                next_action = np.argmax(Q[next_state])

            # Calculate TD error
            td_error = (reward + gamma * Q[next_state, next_action] -
                        Q[state, action])

            # Update eligibility trace for current state-action pair
            E[state, action] += 1

            # Update Q table using TD error and eligibility traces
            Q += alpha * td_error * E

            # Decay eligibility traces
            E *= gamma * lambtha

            # Update state and action
            state, action = next_state, next_action

            # Break if episode ended
            if terminated or truncated:
                break

        # Decay epsilon
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
