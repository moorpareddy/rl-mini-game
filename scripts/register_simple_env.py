"""
Register the SimpleEnv custom environment with Gym.

Run this script before using SimpleEnv-v0 in training.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

from src.envs.simple_env import SimpleEnv


def register_simple_env():
    """Register SimpleEnv as 'SimpleEnv-v0' with Gym."""
    try:
        register(
            id='SimpleEnv-v0',
            entry_point='src.envs.simple_env:SimpleEnv',
            max_episode_steps=50,
        )
        print("SimpleEnv-v0 registered successfully!")
    except Exception as e:
        # Environment might already be registered
        print(f"Note: {e}")


if __name__ == "__main__":
    register_simple_env()
    
    # Test the environment
    print("\nTesting SimpleEnv-v0...")
    env = gym.make('SimpleEnv-v0')
    
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        
        if done:
            print("Episode finished!")
            break
    
    env.close()
    print("\nSimpleEnv-v0 is working correctly!")
