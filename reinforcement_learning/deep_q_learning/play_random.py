# play_random.py - Test si el entorno funciona
import gymnasium as gym
import time

env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84)

for episode in range(2):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = env.action_space.sample()  # Acción aleatoria
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.05)
    
    print(f"Random agent - Episode reward: {total_reward}")

env.close()