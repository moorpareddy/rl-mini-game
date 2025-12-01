"""
Streamlit app for Hugging Face Spaces - Interactive RL Agent Demo
Deploy this to showcase your trained agent online!
"""

import streamlit as st
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time

st.set_page_config(page_title="CartPole RL Agent", page_icon="🎮", layout="wide")

st.title("🎮 CartPole RL Agent - Live Demo")
st.markdown("""
This is a **Proximal Policy Optimization (PPO)** agent trained to balance a pole on a moving cart.
The agent achieved **500/500 perfect score** after training on 50,000 timesteps!
""")

# Sidebar
st.sidebar.header("⚙️ Controls")
num_steps = st.sidebar.slider("Steps to simulate", 10, 500, 100)
run_button = st.sidebar.button("🚀 Run Episode", type="primary")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info("""
**Algorithm:** PPO (Proximal Policy Optimization)  
**Environment:** CartPole-v1  
**Training:** 50,000 timesteps  
**Performance:** 500/500 ✅  
**GitHub:** [View Code](https://github.com/moorpareddy/rl-mini-game)
""")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Agent Performance")
    chart_placeholder = st.empty()

with col2:
    st.subheader("📈 Statistics")
    stats_placeholder = st.empty()

# Load model
@st.cache_resource
def load_model():
    try:
        model = PPO.load("models/best/best_model.zip")
        return model
    except:
        st.error("⚠️ Model not found. Please train the model first.")
        return None

if run_button:
    model = load_model()
    if model:
        env = gym.make("CartPole-v1")
        obs, info = env.reset()
        
        # Storage for visualization
        positions = []
        angles = []
        rewards_list = []
        actions_list = []
        
        progress_bar = st.progress(0)
        
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
            rewards_list.append(reward)
            actions_list.append("→" if action == 1 else "←")
            total_reward += reward
            step += 1
            
            # Update progress
            progress_bar.progress(step / num_steps)
            
            # Update chart every 10 steps
            if step % 10 == 0 or done:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                # Cart position
                ax1.plot(positions, 'b-', linewidth=2)
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                ax1.set_ylabel('Cart Position')
                ax1.set_title('Cart Position Over Time')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(-2.5, 2.5)
                
                # Pole angle
                ax2.plot(angles, 'r-', linewidth=2)
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                ax2.set_ylabel('Pole Angle (degrees)')
                ax2.set_xlabel('Step')
                ax2.set_title('Pole Angle Over Time')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(-15, 15)
                
                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close()
                
                # Update stats
                stats_placeholder.markdown(f"""
                ### Current Stats
                - **Steps:** {step}
                - **Total Reward:** {total_reward}
                - **Current Position:** {cart_pos:.3f}
                - **Current Angle:** {pole_angle * 180 / np.pi:.2f}°
                - **Last Action:** {actions_list[-1]}
                - **Status:** {'✅ Balanced' if not done else '❌ Fell'}
                """)
        
        env.close()
        progress_bar.progress(100)
        
        # Final stats
        if total_reward >= 500:
            st.success("🎉 **Perfect Balance!** Agent achieved maximum score!")
        elif total_reward >= 200:
            st.info(f"👍 **Good Performance!** Score: {total_reward}")
        else:
            st.warning(f"⚠️ **Episode ended early.** Score: {total_reward}")
        
        st.balloons()
else:
    st.info("👈 Click **'Run Episode'** in the sidebar to see the agent in action!")
    
    # Show example image when not running
    st.markdown("### 🎮 How it works")
    st.markdown("""
    1. The **cart** (blue rectangle) moves left or right
    2. The **pole** (red line) must stay balanced upright
    3. The **AI agent** decides which direction to move
    4. Goal: Keep the pole balanced for as long as possible!
    
    The agent learned this behavior through **50,000 practice attempts** using reinforcement learning.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Stable-Baselines3 & Streamlit | 
    <a href='https://github.com/moorpareddy/rl-mini-game'>View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
