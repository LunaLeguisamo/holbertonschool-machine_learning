#!/usr/bin/env python3
"""
Play the trained agent
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    max_steps is the maximum number of steps in the episode
    You need to update 0-load_env.py to add render_mode="ansi"
    Each state of the board should be displayed via the console
    You should always exploit the Q-table
    Ensure that the final state of the environment is also displayed
    after the episode concludes.
    Returns: The total rewards for the episode and a list of rendered
    outputs representing the board state at each step.
    """
    state = env.reset()
    if isinstance(state, tuple):  # Si state es una tupla
        state = state[0]  # Toma el primer elemento
    total_rewards = 0
    rendered_outputs = []
    done = False

    for step in range(max_steps):
        rendered_output = env.render()
        rendered_outputs.append(rendered_output)

        action = np.argmax(Q[state][:])

        res = env.step(action)
        if len(res) == 4:
            new_state, reward, done, _ = res
        elif len(res) == 5:
            new_state, reward, terminated, truncated, _ = res
            done = terminated or truncated

        total_rewards += reward
        state = new_state

        if done:
            break

    # Render the final state
    rendered_output = env.render()
    rendered_outputs.append(rendered_output)

    return total_rewards, rendered_outputs
