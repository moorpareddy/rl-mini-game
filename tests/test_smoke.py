"""
Smoke tests to verify basic functionality.

These tests ensure that the core components work correctly
without running full training sessions.
"""

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from src.envs.simple_env import SimpleEnv


def test_imports():
    """Test that all required libraries can be imported."""
    import torch
    import tensorboard
    import imageio
    import cv2
    
    assert torch is not None
    assert tensorboard is not None
    assert imageio is not None
    assert cv2 is not None


def test_cartpole_env():
    """Test that CartPole environment works."""
    env = gym.make('CartPole-v1')
    obs = env.reset()
    
    assert obs is not None
    assert len(obs) == 4  # CartPole has 4-dimensional state
    
    # Take a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if done:
            break
    
    env.close()


def test_ppo_initialization():
    """Test that PPO can be initialized."""
    env = gym.make('CartPole-v1')
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=42,
    )
    
    assert model is not None
    env.close()


def test_ppo_quick_train():
    """Test a very short training run (50 steps)."""
    env = gym.make('CartPole-v1')
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=42,
    )
    
    # Train for just 50 steps
    model.learn(total_timesteps=50)
    
    # Test prediction
    obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    
    assert action is not None
    assert action in [0, 1]  # CartPole has 2 actions
    
    env.close()


def test_simple_env_creation():
    """Test that SimpleEnv can be created and used."""
    env = SimpleEnv()
    
    obs = env.reset()
    assert obs is not None
    assert len(obs) == 1
    assert 0.0 <= obs[0] <= 1.0
    
    # Take a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert 'distance' in info
        
        if done:
            break
    
    env.close()


def test_simple_env_rendering():
    """Test that SimpleEnv rendering works."""
    env = SimpleEnv()
    env.reset()
    
    # Test rgb_array rendering
    frame = env.render(mode='rgb_array')
    assert frame is not None
    assert frame.shape == (100, 400, 3)
    assert frame.dtype == np.uint8
    
    env.close()


def test_ppo_on_simple_env():
    """Test PPO training on SimpleEnv."""
    env = SimpleEnv()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=42,
    )
    
    # Very short training
    model.learn(total_timesteps=50)
    
    # Test prediction
    obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    
    assert action in [0, 1]
    
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
