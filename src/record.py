"""
Record a video demonstration of a trained PPO agent.

This script loads a trained model and records a video of the agent
performing in the environment. It supports both rgb_array rendering
and RecordVideo wrapper as fallback.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO


def record_video_manual(
    model: PPO,
    env: gym.Env,
    output_path: str,
    n_episodes: int = 1,
    fps: int = 30,
) -> None:
    """
    Record video using manual frame collection with rgb_array rendering.
    
    Args:
        model: Trained PPO model
        env: Gym environment
        output_path: Path to save video file
        n_episodes: Number of episodes to record
        fps: Frames per second for output video
    """
    frames = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            # Render frame
            try:
                frame = env.render()
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Could not render frame: {e}")
            
            # Take action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    # Save video
    if frames:
        print(f"Saving video with {len(frames)} frames to {output_path}")
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved successfully!")
    else:
        print("Warning: No frames were captured!")


def record_video_wrapper(
    model: PPO,
    env_id: str,
    output_path: str,
    n_episodes: int = 1,
) -> None:
    """
    Record video using RecordVideo wrapper (fallback method).
    
    Args:
        model: Trained PPO model
        env_id: Gym environment ID
        output_path: Path to save video file
        n_episodes: Number of episodes to record
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment with RecordVideo wrapper
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        str(output_dir),
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="demo",
    )
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    print(f"Video saved to {output_dir}")


def main(args: argparse.Namespace) -> None:
    """
    Main recording function.
    
    Args:
        args: Command-line arguments
    """
    model_path = Path(args.model)
    output_path = Path(args.output)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Recording Video ===")
    print(f"Model: {model_path}")
    print(f"Environment: {args.env}")
    print(f"Output: {output_path}")
    print(f"Episodes: {args.n_episodes}")
    print()
    
    # Load model
    print("Loading model...")
    model = PPO.load(str(model_path))
    
    # Use RecordVideo wrapper (most reliable method for Gymnasium)
    try:
        print("Using RecordVideo wrapper...")
        record_video_wrapper(
            model,
            args.env,
            str(output_path),
            args.n_episodes,
        )
    except Exception as e:
        print(f"RecordVideo wrapper failed: {e}")
        print("Unable to record video. Please check your gymnasium version and environment.")
        return
    
    print("\n=== Recording complete ===")
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a video demonstration of a trained agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gym environment ID (default: CartPole-v1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demos/demo.mp4",
        help="Output video path (default: demos/demo.mp4)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    
    args = parser.parse_args()
    main(args)
