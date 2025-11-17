#!/usr/bin/env python3
"""Function that trains an agent using"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Train an agent using Monte-Carlo REINFORCE.
    Return the list of scores.
    """
    # CartPole: 4 states, 2 actions
    W = np.random.rand(4, 2)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        score = 0

        while True:
            # Get action and gradient
            action, grad = policy_gradient(state, W)

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store data
            grads.append(grad)
            rewards.append(reward)
            score += reward

            if done:
                break

            state = next_state

        # --- Compute returns ---
        returns = np.zeros_like(rewards, dtype=np.float64)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        # --- Update weights ---
        for grad, Gt in zip(grads, returns):
            W += alpha * Gt * grad

        # Save score
        scores.append(score)

        # Print progress
        print(f"Episode: {episode} Score: {score}")

    return scores
