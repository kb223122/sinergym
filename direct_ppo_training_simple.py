#!/usr/bin/env python3
"""
Direct PPO Agent Training - Simplified Version
==============================================

This script provides a complete PPO (Proximal Policy Optimization) training implementation
using a simple gym environment. Everything is self-contained with no external config files.

WHAT IS PPO?
============
PPO (Proximal Policy Optimization) is a Deep Reinforcement Learning algorithm that:
- Learns to control systems through trial and error
- Uses a neural network to decide actions based on observations
- Gets rewards for good performance and penalties for poor performance
- Balances exploration (trying new actions) with exploitation (using known good actions)

HOW PPO WORKS:
==============
1. OBSERVATION: Agent sees the current state of the environment
2. ACTION: Neural network decides what action to take
3. REWARD: Environment gives feedback (positive/negative reward)
4. LEARNING: Agent updates its policy to get better rewards
5. REPEAT: Process continues until agent learns optimal behavior

PPO ALGORITHM STEPS:
====================
1. Collect experience using current policy
2. Calculate advantages (how much better than expected)
3. Update policy to increase probability of good actions
4. Use "clipping" to prevent too large policy changes
5. Repeat until convergence

This simplified version uses CartPole-v1 environment for demonstration.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import pandas as pd

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    StopTrainingOnNoModelImprovement,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

# Set random seeds for reproducibility
set_random_seed(42)

@dataclass
class PPOTrainingConfig:
    """Configuration class for PPO training parameters."""
    
    # Environment settings
    env_name: str = "CartPole-v1"
    
    # Training parameters
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Evaluation settings
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    eval_log_path: str = "./eval_logs/"
    
    # Model saving
    save_freq: int = 50_000
    model_save_path: str = "./models/"
    
    # Experiment tracking
    experiment_name: str = None
    
    def __post_init__(self):
        """Set default experiment name if not provided."""
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"PPO_{self.env_name}_{timestamp}"

def create_training_environment(config: PPOTrainingConfig) -> gym.Env:
    """
    Create and configure the training environment.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured gym environment
    """
    print(f"🔧 Creating training environment: {config.env_name}")
    
    # Create the environment
    env = gym.make(config.env_name)
    
    # Wrap with Monitor for logging
    env = Monitor(env, filename=f"./logs/{config.experiment_name}_train")
    
    print(f"✅ Environment created successfully")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    
    return env

def create_evaluation_environment(config: PPOTrainingConfig) -> gym.Env:
    """
    Create and configure the evaluation environment.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured gym environment for evaluation
    """
    print(f"🔧 Creating evaluation environment: {config.env_name}")
    
    # Create the environment
    env = gym.make(config.env_name)
    
    # Wrap with Monitor for logging
    env = Monitor(env, filename=f"./logs/{config.experiment_name}_eval")
    
    print(f"✅ Evaluation environment created successfully")
    
    return env

def create_ppo_model(env: gym.Env, config: PPOTrainingConfig) -> PPO:
    """
    Create and configure the PPO model.
    
    Args:
        env: Training environment
        config: Training configuration
        
    Returns:
        Configured PPO model
    """
    print("🤖 Creating PPO model...")
    
    # Create PPO model with specified parameters
    model = PPO(
        "MlpPolicy",  # Use Multi-Layer Perceptron policy
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=1,
        tensorboard_log=None  # Disable tensorboard for simplicity
    )
    
    print("✅ PPO model created successfully")
    print(f"   - Policy network: {model.policy}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Steps per update: {config.n_steps}")
    
    return model

def setup_training_callbacks(config: PPOTrainingConfig, eval_env: gym.Env) -> List[BaseCallback]:
    """
    Set up training callbacks for monitoring and saving.
    
    Args:
        config: Training configuration
        eval_env: Evaluation environment
        
    Returns:
        List of configured callbacks
    """
    print("📊 Setting up training callbacks...")
    
    callbacks = []
    
    # Create directories if they don't exist
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.eval_log_path, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs(f"./tensorboard_logs/{config.experiment_name}", exist_ok=True)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{config.model_save_path}/best/",
        log_path=config.eval_log_path,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=f"{config.model_save_path}/checkpoints/",
        name_prefix=f"{config.experiment_name}_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=10,
        verbose=1
    )
    callbacks.append(early_stopping_callback)
    
    print("✅ Training callbacks configured successfully")
    print(f"   - Evaluation frequency: {config.eval_freq} steps")
    print(f"   - Save frequency: {config.save_freq} steps")
    print(f"   - Evaluation episodes: {config.n_eval_episodes}")
    
    return callbacks

def train_ppo_agent(config: PPOTrainingConfig) -> Tuple[PPO, gym.Env, gym.Env]:
    """
    Train a PPO agent with the specified configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (trained_model, training_env, evaluation_env)
    """
    print("=" * 80)
    print("🚀 STARTING PPO TRAINING PIPELINE")
    print("=" * 80)
    
    try:
        # Phase 1: Environment Setup
        print("\n📦 PHASE 1: ENVIRONMENT SETUP")
        train_env = create_training_environment(config)
        eval_env = create_evaluation_environment(config)
        
        # Phase 2: Model Creation
        print("\n🤖 PHASE 2: MODEL CREATION")
        model = create_ppo_model(train_env, config)
        
        # Phase 3: Callback Setup
        print("\n📊 PHASE 3: CALLBACK SETUP")
        callbacks = setup_training_callbacks(config, eval_env)
        
        # Phase 4: Training
        print("\n🎯 PHASE 4: TRAINING")
        print(f"Training for {config.total_timesteps:,} timesteps...")
        print(f"Expected training time: ~{config.total_timesteps // 10000} minutes")
        
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Phase 5: Final Evaluation
        print("\n📈 PHASE 5: FINAL EVALUATION")
        mean_reward, std_reward = evaluate_policy(
            model, 
            eval_env, 
            n_eval_episodes=20,
            deterministic=True
        )
        
        print(f"Final evaluation results:")
        print(f"   - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   - Training completed successfully!")
        
        return model, train_env, eval_env
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise e

def evaluate_trained_model(model: PPO, eval_env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
    """
    Evaluate a trained model and return performance metrics.
    
    Args:
        model: Trained PPO model
        eval_env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n🔍 Evaluating model over {n_episodes} episodes...")
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=n_episodes,
        deterministic=True
    )
    
    # Run additional episodes to collect detailed metrics
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
            
            if truncated:
                done = True
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': np.mean([r > 195 for r in episode_rewards])  # CartPole success threshold
    }
    
    print("📊 Evaluation Results:")
    print(f"   - Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"   - Reward range: {metrics['min_reward']:.2f} to {metrics['max_reward']:.2f}")
    print(f"   - Mean episode length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    print(f"   - Success rate: {metrics['success_rate']:.1%}")
    
    return metrics

def compare_with_random_policy(eval_env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
    """
    Compare trained model performance with random policy.
    
    Args:
        eval_env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary containing random policy metrics
    """
    print(f"\n🎲 Evaluating random policy over {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = eval_env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
            
            if truncated:
                done = True
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': np.mean([r > 195 for r in episode_rewards])
    }
    
    print("📊 Random Policy Results:")
    print(f"   - Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"   - Reward range: {metrics['min_reward']:.2f} to {metrics['max_reward']:.2f}")
    print(f"   - Mean episode length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    print(f"   - Success rate: {metrics['success_rate']:.1%}")
    
    return metrics

def plot_training_results(train_log_path: str, eval_log_path: str, save_path: str = "./plots/"):
    """
    Plot training and evaluation results.
    
    Args:
        train_log_path: Path to training log file
        eval_log_path: Path to evaluation log file
        save_path: Directory to save plots
    """
    print(f"\n📈 Plotting training results...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Read training logs
    try:
        train_df = pd.read_csv(train_log_path)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Training Results', fontsize=16)
        
        # Training rewards
        if 'r' in train_df.columns:
            axes[0, 0].plot(train_df['l'], train_df['r'])
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Training episode lengths
        if 'l' in train_df.columns:
            axes[0, 1].plot(train_df['l'])
            axes[0, 1].set_title('Training Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True)
        
        # Loss curves (if available)
        if 'train/loss' in train_df.columns:
            axes[1, 0].plot(train_df['train/loss'])
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Value function loss (if available)
        if 'train/value_loss' in train_df.columns:
            axes[1, 1].plot(train_df['train/value_loss'])
            axes[1, 1].set_title('Value Function Loss')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Training plots saved to {save_path}")
        
    except Exception as e:
        print(f"⚠️  Could not plot training results: {e}")

def main():
    """
    Main function to run the complete PPO training pipeline.
    """
    print("=" * 80)
    print("🎯 PPO TRAINING PIPELINE - SIMPLIFIED VERSION")
    print("=" * 80)
    
    # Create configuration
    config = PPOTrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=100_000,  # Reduced for faster demonstration
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        eval_freq=10_000,
        experiment_name="PPO_CartPole_Demo"
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   - Environment: {config.env_name}")
    print(f"   - Total timesteps: {config.total_timesteps:,}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Steps per update: {config.n_steps}")
    print(f"   - Evaluation frequency: {config.eval_freq:,} steps")
    print(f"   - Experiment name: {config.experiment_name}")
    
    # Train the agent
    model, train_env, eval_env = train_ppo_agent(config)
    
    # Evaluate the trained model
    trained_metrics = evaluate_trained_model(model, eval_env, n_episodes=20)
    
    # Compare with random policy
    random_metrics = compare_with_random_policy(eval_env, n_episodes=20)
    
    # Print comparison
    print(f"\n📊 PERFORMANCE COMPARISON")
    print(f"=" * 50)
    print(f"Metric              | Trained Model | Random Policy")
    print(f"=" * 50)
    print(f"Mean Reward         | {trained_metrics['mean_reward']:8.2f} ± {trained_metrics['std_reward']:5.2f} | {random_metrics['mean_reward']:8.2f} ± {random_metrics['std_reward']:5.2f}")
    print(f"Success Rate        | {trained_metrics['success_rate']:8.1%}        | {random_metrics['success_rate']:8.1%}")
    print(f"Mean Episode Length | {trained_metrics['mean_length']:8.1f} ± {trained_metrics['std_length']:5.1f} | {random_metrics['mean_length']:8.1f} ± {random_metrics['std_length']:5.1f}")
    
    # Calculate improvement
    reward_improvement = ((trained_metrics['mean_reward'] - random_metrics['mean_reward']) / random_metrics['mean_reward']) * 100
    print(f"\n🚀 IMPROVEMENT: {reward_improvement:+.1f}% better than random policy!")
    
    # Save the final model
    final_model_path = f"{config.model_save_path}/final_model.zip"
    model.save(final_model_path)
    print(f"\n💾 Final model saved to: {final_model_path}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"🎉 You can now use the trained model for inference!")

if __name__ == "__main__":
    main()