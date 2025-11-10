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
    # Guardar epsilon inicial para el decaimiento
    initial_epsilon = epsilon
    
    for episode in range(episodes):
        # Reset environment and initialize eligibility traces
        state = env.reset()[0]
        E = np.zeros_like(Q)
        
        # Choose initial action with epsilon-greedy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        for step in range(max_steps):
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Choose next action with epsilon-greedy
            if np.random.uniform() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # Calculate TD error - handle terminal state
            if terminated:
                td_error = reward - Q[state, action]
            else:
                td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]
            
            # Update eligibility trace
            E[state, action] += 1
            
            # Update all Q-values
            Q += alpha * td_error * E
            
            # Decay eligibility traces
            E *= gamma * lambtha
            
            # Break if episode ended
            if terminated or truncated:
                break
                
            # Update state and action
            state = next_state
            action = next_action
        
        # Update epsilon (exponential decay)
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
                 np.exp(-epsilon_decay * episode)
    
    return Q