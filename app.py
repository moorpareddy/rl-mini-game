"""
Gradio app for Hugging Face Spaces - Interactive RL Agent Demo
Deploy this to showcase your trained agent online!
"""

import gradio as gr
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load model once at startup
model = None
try:
    model = PPO.load("models/best/best_model.zip")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")

def run_episode(num_steps=100):
    """Run the agent for specified number of steps and return visualization."""
    if model is None:
        return None, "❌ Model not loaded. Please train the model first."
    
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    
    # Storage for visualization
    positions = []
    angles = []
    actions_list = []
    
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < num_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store data
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        positions.append(cart_pos)
        angles.append(pole_angle * 180 / np.pi)  # Convert to degrees
        actions_list.append("→" if action == 1 else "←")
        total_reward += reward
        step += 1
    
    env.close()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Cart position
    ax1.plot(positions, 'b-', linewidth=2, label='Cart Position')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=2.4, color='red', linestyle='--', alpha=0.3, label='Boundary')
    ax1.axhline(y=-2.4, color='red', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Cart Position', fontsize=12)
    ax1.set_title('Cart Position Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-3, 3)
    ax1.legend()
    
    # Pole angle
    ax2.plot(angles, 'r-', linewidth=2, label='Pole Angle')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=12, color='orange', linestyle='--', alpha=0.3, label='Safe Zone')
    ax2.axhline(y=-12, color='orange', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Pole Angle (degrees)', fontsize=12)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_title('Pole Angle Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-20, 20)
    ax2.legend()
    
    plt.tight_layout()
    
    # Create status message
    status = f"""
    ### 📊 Episode Results
    
    **Steps Completed:** {step}/{num_steps}  
    **Total Reward:** {total_reward:.0f}  
    **Final Cart Position:** {positions[-1]:.3f}  
    **Final Pole Angle:** {angles[-1]:.2f}°  
    **Status:** {'✅ Successfully Balanced!' if total_reward >= 200 else '⚠️ Episode Ended Early'}
    
    {'🎉 **Perfect!** Agent achieved maximum score!' if total_reward >= 500 else ''}
    """
    
    return fig, status

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="CartPole RL Agent Demo") as demo:
    gr.Markdown("""
    # 🎮 CartPole RL Agent - Live Demo
    
    This is a **Proximal Policy Optimization (PPO)** agent trained to balance a pole on a moving cart.
    The agent achieved **500/500 perfect score** after training on 50,000 timesteps!
    
    ## 🎯 How it works
    1. The **cart** moves left or right on a track
    2. The **pole** must stay balanced upright
    3. The **AI agent** decides which direction to move each step
    4. Goal: Keep the pole balanced for as long as possible!
    
    The agent learned this behavior through **50,000 practice attempts** using reinforcement learning.
    """)
    
    with gr.Row():
        with gr.Column():
            steps_slider = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                label="Number of Steps to Simulate",
                info="How many steps should the agent run?"
            )
            run_btn = gr.Button("🚀 Run Episode", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("""
            ### 📊 Model Info
            - **Algorithm:** PPO (Proximal Policy Optimization)
            - **Environment:** CartPole-v1
            - **Training Time:** 50,000 timesteps
            - **Performance:** 500/500 ✅
            - **GitHub:** [View Source Code](https://github.com/moorpareddy/rl-mini-game)
            """)
    
    with gr.Row():
        plot_output = gr.Plot(label="Agent Performance Visualization")
    
    with gr.Row():
        status_output = gr.Markdown()
    
    run_btn.click(
        fn=run_episode,
        inputs=[steps_slider],
        outputs=[plot_output, status_output]
    )
    
    gr.Markdown("""
    ---
    ### 💡 What am I seeing?
    - **Top graph (blue):** Shows the cart's position on the track. If it goes outside ±2.4, the episode ends.
    - **Bottom graph (red):** Shows the pole's angle in degrees. The closer to 0°, the more upright the pole is.
    - The agent tries to keep both values within safe ranges to balance the pole!
    
    Built with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and [Gradio](https://gradio.app/)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
