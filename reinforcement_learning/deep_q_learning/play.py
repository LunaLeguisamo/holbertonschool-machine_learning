import gymnasium as gym
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from gymnasium.wrappers import AtariPreprocessing
import time

class GymToOldAPI:
    """
    Wrapper to convert Gymnasium API to old OpenAI Gym API for keras-rl compatibility
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, 'metadata', {'render.modes': ['human', 'rgb_array']})
        
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def render(self, mode='human'):
        # Gymnasium no usa el parámetro 'mode' en render()
        # Pero keras-rl lo pasa, así que lo ignoramos
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)

def create_model(nb_actions, window_length=4):
    """
    Create the same model architecture used during training
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    
    return model

def manual_test(agent, env, nb_episodes=3):
    """
    Manual test function to avoid keras-rl's render issues
    """
    for episode in range(nb_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_reward = 0
        done = False
        steps = 0
        
        print(f"Starting episode {episode + 1}")
        
        while not done:
            # Get action from agent
            action = agent.forward(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            
            # Render the environment
            env.render()
            time.sleep(0.02)  # Slow down for visibility
            
            # Update for next step
            obs = next_obs
            total_reward += reward
            steps += 1
            
            # Print progress
            if steps % 50 == 0:
                print(f"Episode {episode + 1}, Step {steps}, Reward: {total_reward}")
        
        print(f"Episode {episode + 1} finished after {steps} steps with total reward: {total_reward}")
        print("-" * 50)

def main():
    # Create environment with the same preprocessing as training
    print("Creating environment...")
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
    
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False
    )
    
    # Wrap the environment to convert Gymnasium API to old Gym API
    env = GymToOldAPI(env)
    
    # Get action space
    nb_actions = env.action_space.n
    print(f"Number of actions: {nb_actions}")
    
    # Create model with the same architecture
    model = create_model(nb_actions, window_length=4)
    
    # Create agent with greedy policy
    memory = SequentialMemory(limit=100000, window_length=4)
    policy = GreedyQPolicy()
    
    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy
    )
    
    agent.compile(Adam(learning_rate=0.00025))
    
    # Load the trained weights
    try:
        agent.load_weights('policy.h5')
        print("Loaded policy from policy.h5")
        
        # Test the agent using our manual function
        print("Starting gameplay...")
        manual_test(agent, env, nb_episodes=3)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to train the agent first using train.py")
        import traceback
        traceback.print_exc()
    
    env.close()

if __name__ == "__main__":
    main()