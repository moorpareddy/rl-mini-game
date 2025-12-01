"""
Train a PPO agent on a Gym environment with evaluation and checkpointing.

This script provides a complete training pipeline with:
- Reproducible seeding
- TensorBoard logging
- Periodic checkpointing
- Evaluation callbacks
- Model persistence
"""

import argparse
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


def set_seed(seed: int, env: gym.Env) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        env: Gym environment to seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def make_env(env_id: str, log_dir: str, seed: int) -> gym.Env:
    """
    Create a Gym environment wrapped with Monitor for logging.
    
    Args:
        env_id: Gym environment ID (e.g., 'CartPole-v1')
        log_dir: Directory to save monitor logs
        seed: Random seed
        
    Returns:
        Monitored Gym environment
    """
    env = gym.make(env_id)
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    return env


def main(args: argparse.Namespace) -> None:
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Create output directories
    model_dir = Path("models")
    best_model_dir = model_dir / "best"
    checkpoint_dir = model_dir / "checkpoints"
    log_dir = Path(args.log_dir)
    
    model_dir.mkdir(exist_ok=True)
    best_model_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print(f"=== Training PPO on {args.env} ===")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Seed: {args.seed}")
    print(f"Log directory: {log_dir}")
    
    # Create training environment
    train_env = make_env(args.env, str(log_dir / "train"), args.seed)
    set_seed(args.seed, train_env)
    
    # Create separate evaluation environment
    eval_env = make_env(args.env, str(log_dir / "eval"), args.seed + 1)
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        seed=args.seed,
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train the agent
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_run",
    )
    
    # Save final model
    final_model_path = model_dir / "final_model.zip"
    model.save(str(final_model_path))
    print(f"\n=== Training complete ===")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_dir / 'best_model.zip'}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir {log_dir / 'tensorboard'}")
    
    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on a Gym environment"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gym environment ID (default: CartPole-v1)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs (default: logs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    main(args)
