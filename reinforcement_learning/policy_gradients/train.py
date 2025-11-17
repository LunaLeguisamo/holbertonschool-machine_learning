#!/usr/bin/env python3
"""Function that trains an agent using"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Trains a policy gradient model"""
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        done = False
        ep_gradients = []
        rewards = []

        # Render only every 1000 episodes
        if show_result and episode % 1000 == 0:
            env.render()

        while not done:
            action, grad = policy_gradient(state, weight)

            next_state, reward, done, _, _ = env.step(action)

            ep_gradients.append(grad)
            rewards.append(reward)

            state = next_state

        # Monte-Carlo returns Gt
        G = 0
        discounted_rewards = []
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_rewards.insert(0, G)

        # Update weights
        for grad, Gt in zip(ep_gradients, discounted_rewards):
            weight += alpha * grad * Gt

        score = sum(rewards)
        scores.append(score)

        print(f"Episode: {episode} Score: {score}")

    return scores
