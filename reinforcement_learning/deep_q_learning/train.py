import gymnasium as gym
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from gymnasium.wrappers import AtariPreprocessing

class GymToOldAPI:
    """
    Wrapper to convert Gymnasium API to old OpenAI Gym API for keras-rl compatibility
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)

def create_model(nb_actions, window_length=4):
    """
    Create a Deep Q-Network model for Atari Breakout
    """
    model = Sequential()
    
    # Input shape: (window_length, 84, 84)
    model.add(Permute((2, 3, 1), input_shape=(window_length, 84, 84)))
    
    # Convolutional layers
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    
    return model

def main():
    # Create environment with preprocessing
    print("Creating environment...")
    env = gym.make('BreakoutNoFrameskip-v4')
    
    # Apply Atari preprocessing wrappers
    print("Applying preprocessing...")
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
    
    # Create model
    model = create_model(nb_actions, window_length=4)
    print("Model created:")
    print(model.summary())
    
    # Create memory and policy
    memory = SequentialMemory(limit=100000, window_length=4)
    policy = EpsGreedyQPolicy()
    
    # Create agent
    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=10000,
        target_model_update=1000,
        policy=policy,
        gamma=0.99,
        train_interval=4,
        delta_clip=1.0
    )
    
    # Compile the agent
    agent.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Train the agent
    print("Starting training...")
    try:
        history = agent.fit(
            env,
            nb_steps=1000000,
            visualize=False,
            verbose=2,
            log_interval=10000
        )
        
        # Save the policy network
        agent.save_weights('policy.h5', overwrite=True)
        print("Policy saved as policy.h5")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()

if __name__ == "__main__":
    main()