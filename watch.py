"""Watch the trained agent perform in real-time."""

import gymnasium as gym
from stable_baselines3 import PPO
import time

# Load the trained model
print("Loading trained model...")
model = PPO.load("models/best/best_model.zip")

# Create environment with human rendering
env = gym.make("CartPole-v1", render_mode="human")

# Run 5 episodes
num_episodes = 5
print(f"\nWatching agent perform {num_episodes} episodes...")
print("Close the window to stop early.\n")

for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print(f"Episode {episode + 1}/{num_episodes} starting...")
    
    while not done:
        # Render the environment
        env.render()
        
        # Get action from trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        
        # Small delay to make it watchable
        time.sleep(0.02)
    
    print(f"  Completed! Reward: {total_reward}, Steps: {steps}")
    time.sleep(1)  # Pause between episodes

env.close()
print("\nDone! Your agent achieved perfect performance! 🎉")
