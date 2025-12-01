"""
Watch the trained agent with matplotlib visualization (no pygame needed).
This creates an animated visualization showing the cart and pole.
"""

import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation

# Load the trained model
print("Loading trained model...")
model = PPO.load("models/best/best_model.zip")

# Create environment (no rendering needed)
env = gym.make("CartPole-v1")

# Setup matplotlib figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Trained PPO Agent on CartPole-v1', fontsize=16, fontweight='bold')

# For the cart-pole visualization
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-0.5, 2.5)
ax1.set_aspect('equal')
ax1.set_title('Cart-Pole Visualization')
ax1.grid(True, alpha=0.3)

# For the state values plot
ax2.set_xlim(0, 500)
ax2.set_ylim(-3, 3)
ax2.set_title('State Values Over Time')
ax2.set_xlabel('Step')
ax2.set_ylabel('Value')
ax2.grid(True, alpha=0.3)

# Initialize data storage
states_history = {'cart_pos': [], 'cart_vel': [], 'pole_angle': [], 'pole_vel': []}
steps_history = []

# Reset environment
obs, info = env.reset()
done = False
step_count = 0
total_reward = 0

# Cart and pole visual elements
cart = patches.Rectangle((-0.25, 0), 0.5, 0.3, linewidth=2, edgecolor='blue', facecolor='lightblue')
ax1.add_patch(cart)

pole_length = 1.0
pole_line, = ax1.plot([], [], 'r-', linewidth=4, label='Pole')
ax1.plot([-2.4, -2.4, 2.4, 2.4], [0, 0, 0, 0], 'k-', linewidth=2)  # Track
ax1.legend()

# State lines
line_pos, = ax2.plot([], [], 'b-', label='Cart Position', alpha=0.7)
line_angle, = ax2.plot([], [], 'r-', label='Pole Angle', alpha=0.7)
ax2.legend()

info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def init():
    cart.set_xy((-0.25, 0))
    pole_line.set_data([], [])
    line_pos.set_data([], [])
    line_angle.set_data([], [])
    return cart, pole_line, line_pos, line_angle, info_text

def update(frame):
    global obs, done, step_count, total_reward
    
    if done:
        print(f"\nEpisode completed! Steps: {step_count}, Total Reward: {total_reward}")
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        states_history['cart_pos'].clear()
        states_history['cart_vel'].clear()
        states_history['pole_angle'].clear()
        states_history['pole_vel'].clear()
        steps_history.clear()
    
    # Get action from model
    action, _ = model.predict(obs, deterministic=True)
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1
    
    # Extract state values
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    
    # Store history
    states_history['cart_pos'].append(cart_pos)
    states_history['cart_vel'].append(cart_vel)
    states_history['pole_angle'].append(pole_angle)
    states_history['pole_vel'].append(pole_vel)
    steps_history.append(step_count)
    
    # Update cart position
    cart.set_xy((cart_pos - 0.25, 0))
    
    # Update pole position
    pole_x = cart_pos + pole_length * np.sin(pole_angle)
    pole_y = 0.15 + pole_length * np.cos(pole_angle)
    pole_line.set_data([cart_pos, pole_x], [0.15, pole_y])
    
    # Update state plots (show last 200 steps)
    show_steps = min(200, len(steps_history))
    line_pos.set_data(steps_history[-show_steps:], states_history['cart_pos'][-show_steps:])
    line_angle.set_data(steps_history[-show_steps:], states_history['pole_angle'][-show_steps:])
    
    if len(steps_history) > 0:
        ax2.set_xlim(max(0, step_count - 200), max(200, step_count))
    
    # Update info text
    info_text.set_text(f'Step: {step_count}\nReward: {total_reward:.0f}\n'
                       f'Cart Pos: {cart_pos:.3f}\nPole Angle: {pole_angle:.3f}°\n'
                       f'Action: {"RIGHT" if action == 1 else "LEFT"}')
    
    return cart, pole_line, line_pos, line_angle, info_text

print("\nStarting visualization...")
print("The agent will balance the pole indefinitely (max 500 steps per episode).")
print("Close the window to stop.\n")

# Create animation
anim = FuncAnimation(fig, update, init_func=init, frames=None, 
                    interval=20, blit=True, repeat=True)

plt.tight_layout()
plt.show()

env.close()
print("\n✅ Visualization complete!")
