"""
Evaluate a trained PPO agent on a Gym environment.

This script loads a trained model and runs deterministic episodes
to compute mean and standard deviation of episode rewards.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate_model(
    model: PPO,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> tuple[float, float, list[float]]:
    """
    Evaluate a trained model over multiple episodes.
    
    Args:
        model: Trained PPO model
        env: Gym environment
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        
    Returns:
        Tuple of (mean_reward, std_reward, all_rewards)
    """
    all_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        all_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    return mean_reward, std_reward, all_rewards


def main(args: argparse.Namespace) -> None:
    """
    Main evaluation function.
    
    Args:
        args: Command-line arguments
    """
    model_path = Path(args.model)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"=== Evaluating Model ===")
    print(f"Model: {model_path}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print()
    
    # Load model
    print("Loading model...")
    model = PPO.load(str(model_path))
    
    # Create environment
    env = gym.make(args.env)
    
    # Evaluate
    print(f"\nRunning {args.n_episodes} evaluation episodes...\n")
    mean_reward, std_reward, all_rewards = evaluate_model(
        model, env, args.n_episodes, args.deterministic
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min(all_rewards):.2f}")
    print(f"Max reward: {max(all_rewards):.2f}")
    print(f"All rewards: {[f'{r:.2f}' for r in all_rewards]}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent"
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
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    
    args = parser.parse_args()
    main(args)
