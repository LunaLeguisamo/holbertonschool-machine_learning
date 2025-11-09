#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function V(s).

    Parameters:
        env: the environment instance (e.g., FrozenLake8x8-v1)
        V: numpy.ndarray of shape (n_states,) containing value estimates
        policy: function that takes in a state and returns an action
        episodes: number of episodes to run
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value function
    """

    # Desc (mapa del entorno) → para saber dónde hay agujeros (b'H')
    desc = env.unwrapped.desc
    n_rows, n_cols = desc.shape

    for ep in range(episodes):
        # Reiniciamos el entorno
        state, _ = env.reset()

        # Guardamos trayectorias
        states = [state]
        rewards = []

        terminated = False
        truncated = False

        # Jugar un episodio completo (hasta que termina o alcanza max_steps)
        for t in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(next_state)
            rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break

        # Recorrer el episodio hacia atrás (para calcular retornos G)
        G = 0.0
        visited = set()  # para first-visit Monte Carlo

        for i in range(len(states) - 2, -1, -1):
            s = states[i]
            r = rewards[i]
            G = r + gamma * G  # retorno descontado acumulado

            # Evitar múltiples actualizaciones del mismo estado en un episodio
            if s in visited:
                continue
            visited.add(s)

            # No actualizar agujeros (mantener valor -1)
            row, col = divmod(s, n_cols)
            if desc[row, col] == b'H':
                continue

            # Actualización incremental del valor
            V[s] = V[s] + alpha * (G - V[s])

    return V
