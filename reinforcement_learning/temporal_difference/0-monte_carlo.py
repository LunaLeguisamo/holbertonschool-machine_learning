#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function.
    """
    n_states = V.shape[0]
    desc = env.unwrapped.desc
    n_rows, n_cols = desc.shape

    for ep in range(episodes):
        state, _ = env.reset()

        states = []
        rewards = []
        done = False

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            state = next_state
            done = terminated or truncated
            if done:
                break

        # Calculate returns backward
        G = 0.0
        visited = set()

        for i in range(len(states) - 1, -1, -1):
            s = states[i]
            r = rewards[i]
            G = r + gamma * G

            if s in visited:
                continue
            visited.add(s)

            # No updates for holes
            row, col = divmod(s, n_cols)
            if desc[row, col] == b'H':
                continue

            V[s] += alpha * (G - V[s])

    return V
