"""
A minimal custom Gym environment example.

This environment demonstrates how to create a custom environment
that works with Stable-Baselines3 and supports both 'rgb_array'
and 'human' rendering modes.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimpleEnv(gym.Env):
    """
    A simple 1D environment where the agent tries to reach a target position.
    
    State: Single float representing position in [0, 1]
    Actions: 0 = move left, 1 = move right
    Goal: Reach position close to 0.7
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        """Initialize the environment."""
        super(SimpleEnv, self).__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Environment parameters
        self.target = 0.7
        self.step_size = 0.1
        self.max_steps = 50
        
        # State variables
        self.position = None
        self.steps = 0
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.position = np.random.uniform(0.0, 1.0)
        self.steps = 0
        return np.array([self.position], dtype=np.float32)
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (move left) or 1 (move right)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Move position based on action
        if action == 0:  # Move left
            self.position -= self.step_size
        else:  # Move right
            self.position += self.step_size
        
        # Clip position to valid range
        self.position = np.clip(self.position, 0.0, 1.0)
        
        # Calculate reward (negative distance to target)
        distance = abs(self.position - self.target)
        reward = -distance
        
        # Check if episode is done
        self.steps += 1
        done = self.steps >= self.max_steps or distance < 0.05
        
        # Bonus reward for reaching target
        if distance < 0.05:
            reward += 10.0
        
        obs = np.array([self.position], dtype=np.float32)
        info = {'distance': distance, 'steps': self.steps}
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: 'human' for text output, 'rgb_array' for image
            
        Returns:
            RGB array if mode='rgb_array', None otherwise
        """
        if mode == 'human':
            # Text-based rendering
            bar_length = 50
            pos_idx = int(self.position * bar_length)
            target_idx = int(self.target * bar_length)
            
            bar = ['-'] * bar_length
            bar[target_idx] = 'T'  # Target
            bar[pos_idx] = 'A'  # Agent
            
            print(f"Step {self.steps}: [{''.join(bar)}] Pos={self.position:.2f}")
            
        elif mode == 'rgb_array':
            # Create a simple RGB image
            width, height = 400, 100
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Draw target (green line)
            target_x = int(self.target * width)
            img[:, max(0, target_x-2):min(width, target_x+2), :] = [0, 255, 0]
            
            # Draw agent (red circle)
            agent_x = int(self.position * width)
            agent_y = height // 2
            
            # Simple circle drawing
            for dy in range(-10, 11):
                for dx in range(-10, 11):
                    if dx*dx + dy*dy <= 100:  # Circle radius = 10
                        y = agent_y + dy
                        x = agent_x + dx
                        if 0 <= y < height and 0 <= x < width:
                            img[y, x, :] = [255, 0, 0]
            
            return img
        
        return None
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed=None):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
            
        Returns:
            List containing the seed
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]
