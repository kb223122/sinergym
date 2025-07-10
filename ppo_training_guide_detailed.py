#!/usr/bin/env python3
"""
PPO Training and Evaluation Guide - Detailed Beginner-Friendly Tutorial
======================================================================

This script provides a comprehensive guide to train and evaluate a PPO (Proximal Policy Optimization)
agent for the Sinergym 5Zone environment. Every line is explained in detail for beginners.

What is PPO?
- PPO is a Deep Reinforcement Learning algorithm
- It learns to control HVAC systems by trial and error
- The agent observes the building state and decides on heating/cooling setpoints
- It gets rewards for balancing energy efficiency and occupant comfort

Author: AI Assistant
Date: 2024
"""

# ============================================================================
# IMPORTS - All the libraries we need
# ============================================================================

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Gymnasium - The standard RL environment interface
import gymnasium as gym

# Sinergym - Building simulation environment
import sinergym

# Stable Baselines 3 - RL algorithms library (includes PPO)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Sinergym utilities
from sinergym.utils.wrappers import (
    NormalizeObservation,    # Scales observations to help training
    NormalizeAction,         # Scales actions to help training  
    LoggerWrapper,           # Logs environment interactions
    CSVLogger               # Saves data to CSV files
)
from sinergym.utils.callbacks import LoggerEvalCallback


# ============================================================================
# CONFIGURATION CLASS - Store all our training settings
# ============================================================================

class PPOConfig:
    """
    Configuration class to store all PPO training parameters.
    Think of this as a settings file for our AI training.
    """
    def __init__(self):
        # Environment settings
        self.env_name = 'Eplus-5zone-hot-continuous-v1'  # Which building simulation to use
        self.experiment_name = f"PPO_5Zone_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Training parameters - these control how the AI learns
        self.total_timesteps = 100000      # How many steps to train (more = better learning, longer time)
        self.eval_freq = 10000             # How often to test the agent during training
        self.n_eval_episodes = 3           # Number of episodes to test each time
        
        # PPO hyperparameters - these are the "knobs" that control how PPO learns
        # Don't worry about understanding all of these initially
        self.learning_rate = 3e-4          # How fast the AI learns (too high = unstable, too low = slow)
        self.n_steps = 2048                # Steps collected before each update
        self.batch_size = 64               # Size of data batches for training
        self.n_epochs = 10                 # How many times to reuse each batch of data
        self.gamma = 0.99                  # How much future rewards matter (0-1)
        self.gae_lambda = 0.95             # Smoothing parameter for advantage estimation
        self.clip_range = 0.2              # Prevents too large policy updates
        self.ent_coef = 0.01               # Encourages exploration
        self.vf_coef = 0.5                 # Value function loss coefficient
        
        # Evaluation settings
        self.final_eval_episodes = 5       # Episodes for final evaluation
        
        # Output directories
        self.model_save_path = "./models/"
        self.logs_path = "./logs/"
        self.plots_path = "./plots/"
        
        # Create directories if they don't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
    
    def save_config(self, filepath: str):
        """Save configuration to a JSON file for reference."""
        config_dict = {
            'env_name': self.env_name,
            'experiment_name': self.experiment_name,
            'total_timesteps': self.total_timesteps,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to: {filepath}")


# ============================================================================
# ENVIRONMENT CREATION FUNCTIONS
# ============================================================================

def create_training_environment(config: PPOConfig) -> gym.Env:
    """
    Create and configure the training environment with all necessary wrappers.
    
    Wrappers are like "filters" that modify how the environment behaves:
    - NormalizeObservation: Scales sensor readings to similar ranges (helps AI learn)
    - NormalizeAction: Scales actions to a standard range
    - LoggerWrapper: Records what happens during training
    - CSVLogger: Saves data to files we can analyze later
    
    Args:
        config: Configuration object with all our settings
        
    Returns:
        gym.Env: Wrapped environment ready for training
    """
    print(f"Creating training environment: {config.env_name}")
    
    # Step 1: Create the base environment
    # This is the actual building simulation
    env = gym.make(config.env_name, env_name=f"{config.experiment_name}_TRAIN")
    
    print(f"Base environment created:")
    print(f"  - Observation space: {env.observation_space}")  # What the AI can observe
    print(f"  - Action space: {env.action_space}")            # What actions AI can take
    print(f"  - Episode length: {env.timestep_per_episode} timesteps")  # How long each episode is
    
    # Step 2: Apply normalization wrappers
    # These help the AI learn by putting all numbers in similar ranges
    print("Applying normalization wrappers...")
    
    # Normalize observations (sensor readings) to have mean=0, std=1
    # This helps because AI algorithms work better when all inputs are in similar ranges
    env = NormalizeObservation(env)
    
    # Normalize actions to range [-1, 1]
    # PPO works best when actions are in this standard range
    env = NormalizeAction(env)
    
    # Step 3: Apply logging wrappers
    # These record everything that happens so we can analyze it later
    print("Applying logging wrappers...")
    
    # LoggerWrapper records environment interactions
    env = LoggerWrapper(env)
    
    # CSVLogger saves data to CSV files for analysis
    env = CSVLogger(env)
    
    print(f"Training environment ready!")
    print(f"  - Final observation space: {env.observation_space}")
    print(f"  - Final action space: {env.action_space}")
    
    return env


def create_evaluation_environment(config: PPOConfig, mean=None, var=None) -> gym.Env:
    """
    Create an evaluation environment for testing the trained agent.
    
    This is similar to the training environment but:
    - Uses fixed normalization parameters (mean/var) from training
    - Disables learning-related updates
    - Used only for testing, not training
    
    Args:
        config: Configuration object
        mean: Mean values for observation normalization (from training)
        var: Variance values for observation normalization (from training)
        
    Returns:
        gym.Env: Environment configured for evaluation
    """
    print(f"Creating evaluation environment: {config.env_name}")
    
    # Create base environment
    env = gym.make(config.env_name, env_name=f"{config.experiment_name}_EVAL")
    
    # Apply same wrappers as training, but with fixed normalization
    if mean is not None and var is not None:
        # Use the same normalization parameters as training
        # automatic_update=False means these parameters won't change during evaluation
        env = NormalizeObservation(env, mean=mean, var=var, automatic_update=False)
    else:
        # If we don't have training parameters, use default normalization
        env = NormalizeObservation(env)
    
    env = NormalizeAction(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)
    
    print("Evaluation environment ready!")
    return env


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_ppo_model(env: gym.Env, config: PPOConfig) -> PPO:
    """
    Create a PPO model with our specified hyperparameters.
    
    PPO (Proximal Policy Optimization) is the AI algorithm that will learn to control the building.
    Think of it as creating a "brain" that will learn through trial and error.
    
    Args:
        env: The environment the model will learn in
        config: Configuration with all the learning parameters
        
    Returns:
        PPO: Configured PPO model ready for training
    """
    print("Creating PPO model...")
    
    # Create the PPO model
    # 'MlpPolicy' means we use a Multi-Layer Perceptron (neural network) for the AI brain
    model = PPO(
        policy='MlpPolicy',           # Type of neural network architecture
        env=env,                      # Environment to learn in
        learning_rate=config.learning_rate,    # How fast to learn
        n_steps=config.n_steps,               # Steps to collect before each update
        batch_size=config.batch_size,         # Size of training batches
        n_epochs=config.n_epochs,             # How many times to reuse each batch
        gamma=config.gamma,                   # Discount factor for future rewards
        gae_lambda=config.gae_lambda,         # GAE parameter for advantage estimation
        clip_range=config.clip_range,         # PPO clipping parameter
        ent_coef=config.ent_coef,             # Entropy coefficient (encourages exploration)
        vf_coef=config.vf_coef,               # Value function coefficient
        verbose=1,                            # Print training progress
        device='auto'                         # Use GPU if available, otherwise CPU
    )
    
    print("PPO model created with parameters:")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Training epochs per update: {config.n_epochs}")
    print(f"  - Steps per update: {config.n_steps}")
    print(f"  - Device: {model.device}")
    
    return model


def setup_training_callbacks(config: PPOConfig, eval_env: gym.Env) -> CallbackList:
    """
    Set up callbacks for monitoring and saving during training.
    
    Callbacks are functions that run at specific times during training:
    - They can save the best model
    - Log performance metrics
    - Stop training early if needed
    
    Args:
        config: Configuration object
        eval_env: Environment for evaluation during training
        
    Returns:
        CallbackList: List of callbacks to use during training
    """
    print("Setting up training callbacks...")
    
    callbacks = []
    
    # Evaluation callback - periodically tests the agent and saves the best model
    eval_callback = EvalCallback(
        eval_env=eval_env,                           # Environment for testing
        best_model_save_path=config.model_save_path, # Where to save the best model
        log_path=config.logs_path,                   # Where to save evaluation logs
        eval_freq=config.eval_freq,                  # How often to evaluate
        n_eval_episodes=config.n_eval_episodes,     # Episodes per evaluation
        deterministic=True,                          # Use deterministic actions for evaluation
        render=False,                                # Don't render during evaluation
        verbose=1                                    # Print evaluation results
    )
    callbacks.append(eval_callback)
    
    print(f"Evaluation callback configured:")
    print(f"  - Evaluation frequency: every {config.eval_freq} timesteps")
    print(f"  - Episodes per evaluation: {config.n_eval_episodes}")
    print(f"  - Best model save path: {config.model_save_path}")
    
    return CallbackList(callbacks)


def train_ppo_agent(config: PPOConfig) -> Tuple[PPO, gym.Env, gym.Env]:
    """
    Complete PPO training pipeline.
    
    This function handles the entire training process:
    1. Create environments
    2. Create PPO model
    3. Set up monitoring
    4. Train the model
    5. Save everything
    
    Args:
        config: Configuration with all training parameters
        
    Returns:
        Tuple[PPO, gym.Env, gym.Env]: Trained model, training env, evaluation env
    """
    print("="*80)
    print("STARTING PPO TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Create training environment
    print("\n1. Creating Training Environment")
    print("-" * 40)
    train_env = create_training_environment(config)
    
    # Step 2: Create evaluation environment
    print("\n2. Creating Evaluation Environment")
    print("-" * 40)
    eval_env = create_evaluation_environment(config)
    
    # Step 3: Create PPO model
    print("\n3. Creating PPO Model")
    print("-" * 40)
    model = create_ppo_model(train_env, config)
    
    # Step 4: Set up callbacks for monitoring
    print("\n4. Setting up Training Callbacks")
    print("-" * 40)
    callbacks = setup_training_callbacks(config, eval_env)
    
    # Step 5: Configure logging
    print("\n5. Configuring Training Logs")
    print("-" * 40)
    model.set_logger(configure(config.logs_path, ["stdout", "csv", "tensorboard"]))
    
    # Step 6: Save configuration
    config_path = os.path.join(config.logs_path, "training_config.json")
    config.save_config(config_path)
    
    # Step 7: Start training!
    print("\n6. Starting Training")
    print("-" * 40)
    print(f"Training for {config.total_timesteps:,} timesteps...")
    print("This may take a while depending on your computer and the number of timesteps.")
    print("You can monitor progress in the terminal output.\n")
    
    # The actual training happens here
    # The model will learn by repeatedly:
    # 1. Observing the building state
    # 2. Choosing actions (setpoints)
    # 3. Receiving rewards
    # 4. Updating its policy to get better rewards
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,          # Show progress bar
        log_interval=1              # Log every update
    )
    
    # Step 8: Save the final model
    print("\n7. Saving Final Model")
    print("-" * 40)
    final_model_path = os.path.join(config.model_save_path, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return model, train_env, eval_env


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_trained_model(model: PPO, eval_env: gym.Env, config: PPOConfig) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the trained PPO model.
    
    This function tests how well our trained AI performs by:
    - Running several complete episodes
    - Recording all performance metrics
    - Analyzing energy consumption and comfort
    
    Args:
        model: Trained PPO model
        eval_env: Environment for evaluation
        config: Configuration object
        
    Returns:
        Dict: Comprehensive evaluation results
    """
    print("\n" + "="*80)
    print("EVALUATING TRAINED PPO MODEL")
    print("="*80)
    
    results = {
        'episode_rewards': [],
        'episode_energies': [],
        'episode_comfort_violations': [],
        'episode_lengths': [],
        'monthly_data': []
    }
    
    print(f"Running {config.final_eval_episodes} evaluation episodes...")
    
    for episode in range(config.final_eval_episodes):
        print(f"\nEpisode {episode + 1}/{config.final_eval_episodes}")
        print("-" * 40)
        
        # Reset environment for new episode
        obs, info = eval_env.reset()
        
        # Initialize episode tracking variables
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        episode_length = 0
        monthly_rewards = []
        current_month = info.get('month', 1)
        monthly_reward = 0
        
        # Run one complete episode
        terminated = truncated = False
        
        while not (terminated or truncated):
            # Get action from trained model
            # deterministic=True means we use the best action (no exploration)
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Track metrics
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            episode_length += 1
            monthly_reward += reward
            
            # Check if month changed (for monthly analysis)
            if info.get('month', 1) != current_month:
                monthly_rewards.append({
                    'month': current_month,
                    'reward': monthly_reward,
                    'energy': info.get('total_power_demand', 0),
                    'comfort': info.get('total_temperature_violation', 0)
                })
                current_month = info.get('month', 1)
                monthly_reward = 0
        
        # Store episode results
        results['episode_rewards'].append(episode_reward)
        results['episode_energies'].append(episode_energy)
        results['episode_comfort_violations'].append(episode_comfort_violation)
        results['episode_lengths'].append(episode_length)
        results['monthly_data'].append(monthly_rewards)
        
        # Print episode summary
        print(f"Episode completed:")
        print(f"  - Total reward: {episode_reward:.2f}")
        print(f"  - Total energy: {episode_energy:.2f} kWh")
        print(f"  - Comfort violations: {episode_comfort_violation:.2f} °C⋅hours")
        print(f"  - Episode length: {episode_length} timesteps")
    
    # Calculate summary statistics
    results['mean_reward'] = np.mean(results['episode_rewards'])
    results['std_reward'] = np.std(results['episode_rewards'])
    results['mean_energy'] = np.mean(results['episode_energies'])
    results['std_energy'] = np.std(results['episode_energies'])
    results['mean_comfort_violation'] = np.mean(results['episode_comfort_violations'])
    results['std_comfort_violation'] = np.std(results['episode_comfort_violations'])
    results['mean_episode_length'] = np.mean(results['episode_lengths'])
    
    # Print summary
    print(f"\nEVALUATION SUMMARY")
    print("-" * 40)
    print(f"Episodes evaluated: {config.final_eval_episodes}")
    print(f"Mean episode reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean energy consumption: {results['mean_energy']:.2f} ± {results['std_energy']:.2f} kWh")
    print(f"Mean comfort violations: {results['mean_comfort_violation']:.2f} ± {results['std_comfort_violation']:.2f} °C⋅hours")
    print(f"Mean episode length: {results['mean_episode_length']:.0f} timesteps")
    
    return results


def compare_with_random_policy(eval_env: gym.Env, config: PPOConfig) -> Dict[str, Any]:
    """
    Evaluate a random policy for comparison with the trained model.
    
    This helps us understand how much better our trained AI is compared to
    just taking random actions (which is a very basic baseline).
    
    Args:
        eval_env: Environment for evaluation
        config: Configuration object
        
    Returns:
        Dict: Random policy evaluation results
    """
    print("\n" + "="*60)
    print("EVALUATING RANDOM POLICY (BASELINE)")
    print("="*60)
    
    results = {
        'episode_rewards': [],
        'episode_energies': [],
        'episode_comfort_violations': [],
        'episode_lengths': []
    }
    
    print(f"Running {config.final_eval_episodes} random policy episodes...")
    
    for episode in range(config.final_eval_episodes):
        print(f"\nRandom Episode {episode + 1}/{config.final_eval_episodes}")
        
        # Reset environment
        obs, info = eval_env.reset()
        
        # Initialize tracking variables
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        episode_length = 0
        
        # Run episode with random actions
        terminated = truncated = False
        
        while not (terminated or truncated):
            # Take random action (no intelligence here!)
            action = eval_env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Track metrics
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            episode_length += 1
        
        # Store results
        results['episode_rewards'].append(episode_reward)
        results['episode_energies'].append(episode_energy)
        results['episode_comfort_violations'].append(episode_comfort_violation)
        results['episode_lengths'].append(episode_length)
        
        print(f"  Random episode reward: {episode_reward:.2f}")
    
    # Calculate summary statistics
    results['mean_reward'] = np.mean(results['episode_rewards'])
    results['std_reward'] = np.std(results['episode_rewards'])
    results['mean_energy'] = np.mean(results['episode_energies'])
    results['std_energy'] = np.std(results['episode_energies'])
    results['mean_comfort_violation'] = np.mean(results['episode_comfort_violations'])
    results['std_comfort_violation'] = np.std(results['episode_comfort_violations'])
    results['mean_episode_length'] = np.mean(results['episode_lengths'])
    
    print(f"\nRANDOM POLICY SUMMARY")
    print("-" * 30)
    print(f"Mean episode reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean energy consumption: {results['mean_energy']:.2f} ± {results['std_energy']:.2f} kWh")
    print(f"Mean comfort violations: {results['mean_comfort_violation']:.2f} ± {results['std_comfort_violation']:.2f} °C⋅hours")
    
    return results


def create_performance_plots(trained_results: Dict, random_results: Dict, config: PPOConfig):
    """
    Create visualization plots comparing trained model vs random policy.
    
    Visual plots help us understand:
    - How much better our AI is compared to random actions
    - Which aspects (energy, comfort) improved the most
    - Consistency of performance across episodes
    
    Args:
        trained_results: Results from trained model evaluation
        random_results: Results from random policy evaluation
        config: Configuration object
    """
    print("\nCreating performance comparison plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPO Model vs Random Policy Performance Comparison', fontsize=16)
    
    # Plot 1: Episode Rewards Comparison
    axes[0, 0].boxplot([trained_results['episode_rewards'], random_results['episode_rewards']], 
                       labels=['Trained PPO', 'Random Policy'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_ylabel('Total Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Energy Consumption Comparison
    axes[0, 1].boxplot([trained_results['episode_energies'], random_results['episode_energies']], 
                       labels=['Trained PPO', 'Random Policy'])
    axes[0, 1].set_title('Energy Consumption')
    axes[0, 1].set_ylabel('Total Energy (kWh)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Comfort Violations Comparison
    axes[1, 0].boxplot([trained_results['episode_comfort_violations'], random_results['episode_comfort_violations']], 
                       labels=['Trained PPO', 'Random Policy'])
    axes[1, 0].set_title('Comfort Violations')
    axes[1, 0].set_ylabel('Total Violations (°C⋅hours)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance Metrics Bar Chart
    metrics = ['Reward', 'Energy', 'Comfort']
    trained_means = [trained_results['mean_reward'], trained_results['mean_energy'], trained_results['mean_comfort_violation']]
    random_means = [random_results['mean_reward'], random_results['mean_energy'], random_results['mean_comfort_violation']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, trained_means, width, label='Trained PPO', alpha=0.8)
    axes[1, 1].bar(x + width/2, random_means, width, label='Random Policy', alpha=0.8)
    axes[1, 1].set_title('Mean Performance Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config.plots_path, 'performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance plots saved to: {plot_path}")


def generate_final_report(trained_results: Dict, random_results: Dict, config: PPOConfig):
    """
    Generate a comprehensive final report with all results and insights.
    
    This creates a detailed text report that summarizes:
    - Training configuration used
    - Performance improvements achieved
    - Recommendations for further improvements
    
    Args:
        trained_results: Results from trained model evaluation
        random_results: Results from random policy evaluation
        config: Configuration object
    """
    print("\nGenerating final evaluation report...")
    
    report_path = os.path.join(config.logs_path, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("PPO TRAINING AND EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Training Configuration
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 25 + "\n")
        f.write(f"Environment: {config.env_name}\n")
        f.write(f"Total Training Timesteps: {config.total_timesteps:,}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Training Epochs: {config.n_epochs}\n")
        f.write(f"Evaluation Episodes: {config.final_eval_episodes}\n\n")
        
        # Performance Comparison
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 25 + "\n")
        
        # Calculate improvements
        reward_improvement = trained_results['mean_reward'] - random_results['mean_reward']
        energy_improvement = ((random_results['mean_energy'] - trained_results['mean_energy']) / 
                             random_results['mean_energy']) * 100
        comfort_improvement = ((random_results['mean_comfort_violation'] - trained_results['mean_comfort_violation']) / 
                              random_results['mean_comfort_violation']) * 100
        
        f.write(f"{'Metric':<25} {'Trained PPO':<15} {'Random Policy':<15} {'Improvement':<15}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Mean Reward':<25} {trained_results['mean_reward']:<15.2f} {random_results['mean_reward']:<15.2f} {reward_improvement:<15.2f}\n")
        f.write(f"{'Mean Energy (kWh)':<25} {trained_results['mean_energy']:<15.2f} {random_results['mean_energy']:<15.2f} {energy_improvement:<15.1f}%\n")
        f.write(f"{'Mean Comfort Viol.':<25} {trained_results['mean_comfort_violation']:<15.2f} {random_results['mean_comfort_violation']:<15.2f} {comfort_improvement:<15.1f}%\n\n")
        
        # Detailed Statistics
        f.write("DETAILED STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write("Trained PPO Model:\n")
        f.write(f"  Reward: {trained_results['mean_reward']:.2f} ± {trained_results['std_reward']:.2f}\n")
        f.write(f"  Energy: {trained_results['mean_energy']:.2f} ± {trained_results['std_energy']:.2f} kWh\n")
        f.write(f"  Comfort: {trained_results['mean_comfort_violation']:.2f} ± {trained_results['std_comfort_violation']:.2f} °C⋅hours\n\n")
        
        f.write("Random Policy:\n")
        f.write(f"  Reward: {random_results['mean_reward']:.2f} ± {random_results['std_reward']:.2f}\n")
        f.write(f"  Energy: {random_results['mean_energy']:.2f} ± {random_results['std_energy']:.2f} kWh\n")
        f.write(f"  Comfort: {random_results['mean_comfort_violation']:.2f} ± {random_results['std_comfort_violation']:.2f} °C⋅hours\n\n")
        
        # Insights and Recommendations
        f.write("INSIGHTS AND RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        
        if reward_improvement > 0:
            f.write("✓ The trained PPO model successfully learned to achieve higher rewards.\n")
        else:
            f.write("⚠ The trained model did not achieve higher rewards than random policy.\n")
            f.write("  Consider: longer training, different hyperparameters, or reward function tuning.\n")
        
        if energy_improvement > 0:
            f.write(f"✓ Energy consumption improved by {energy_improvement:.1f}%.\n")
        else:
            f.write("⚠ Energy consumption did not improve. Consider increasing energy weight in reward function.\n")
        
        if comfort_improvement > 0:
            f.write(f"✓ Comfort violations reduced by {comfort_improvement:.1f}%.\n")
        else:
            f.write("⚠ Comfort violations did not improve. Consider adjusting comfort ranges or temperature weight.\n")
        
        f.write("\nFor further improvements, consider:\n")
        f.write("- Longer training (more timesteps)\n")
        f.write("- Hyperparameter tuning\n")
        f.write("- Different reward function weights\n")
        f.write("- Environment wrapper modifications\n")
        f.write("- Advanced PPO configurations\n")
    
    print(f"Final report saved to: {report_path}")


# ============================================================================
# MODEL LOADING AND EVALUATION FUNCTIONS
# ============================================================================

def load_and_evaluate_model(model_path: str, config: PPOConfig) -> Tuple[PPO, Dict[str, Any]]:
    """
    Load a previously trained model and evaluate it.
    
    This function is useful when you want to:
    - Test a model you trained earlier
    - Compare different trained models
    - Use a pre-trained model for deployment
    
    Args:
        model_path: Path to the saved model file
        config: Configuration object
        
    Returns:
        Tuple[PPO, Dict]: Loaded model and evaluation results
    """
    print(f"\n" + "="*60)
    print("LOADING AND EVALUATING SAVED MODEL")
    print("="*60)
    
    # Check if model file exists
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model file not found: {model_path}.zip")
    
    print(f"Loading model from: {model_path}")
    
    # Load the trained model
    # Stable Baselines 3 automatically saves models as .zip files
    model = PPO.load(model_path)
    
    print("Model loaded successfully!")
    print(f"  Policy architecture: {type(model.policy).__name__}")
    print(f"  Device: {model.device}")
    
    # Create evaluation environment
    # Note: We need to use the same wrappers as during training
    eval_env = create_evaluation_environment(config)
    
    # Evaluate the loaded model
    results = evaluate_trained_model(model, eval_env, config)
    
    # Clean up
    eval_env.close()
    
    return model, results


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_complete_training_pipeline():
    """
    Run the complete PPO training and evaluation pipeline.
    
    This is the main function that orchestrates everything:
    1. Set up configuration
    2. Train the model
    3. Evaluate the trained model
    4. Compare with random policy
    5. Generate reports and plots
    """
    print("PPO TRAINING AND EVALUATION PIPELINE")
    print("=" * 50)
    print("This script will train a PPO agent to control building HVAC systems.")
    print("The training process includes:")
    print("- Environment setup with proper wrappers")
    print("- PPO model creation and training")
    print("- Periodic evaluation during training")
    print("- Final comprehensive evaluation")
    print("- Comparison with random policy baseline")
    print("- Performance visualization and reporting")
    print()
    
    # Step 1: Create configuration
    config = PPOConfig()
    print(f"Experiment name: {config.experiment_name}")
    print(f"Training for {config.total_timesteps:,} timesteps")
    print()
    
    try:
        # Step 2: Train the model
        trained_model, train_env, eval_env = train_ppo_agent(config)
        
        # Step 3: Evaluate trained model
        print("\nEvaluating trained model...")
        trained_results = evaluate_trained_model(trained_model, eval_env, config)
        
        # Step 4: Compare with random policy
        print("\nComparing with random policy...")
        random_results = compare_with_random_policy(eval_env, config)
        
        # Step 5: Create visualizations
        create_performance_plots(trained_results, random_results, config)
        
        # Step 6: Generate final report
        generate_final_report(trained_results, random_results, config)
        
        # Step 7: Clean up
        train_env.close()
        eval_env.close()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"All outputs saved in:")
        print(f"  - Models: {config.model_save_path}")
        print(f"  - Logs: {config.logs_path}")
        print(f"  - Plots: {config.plots_path}")
        
        return trained_model, trained_results, random_results
        
    except Exception as e:
        print(f"\nError during training pipeline: {e}")
        print("Please check your environment setup and try again.")
        raise


def run_evaluation_only(model_path: str):
    """
    Run evaluation only for a previously trained model.
    
    Use this function when you already have a trained model and just want to
    evaluate its performance without training again.
    
    Args:
        model_path: Path to the saved model (without .zip extension)
    """
    print("PPO MODEL EVALUATION ONLY")
    print("=" * 30)
    
    # Create configuration for evaluation
    config = PPOConfig()
    
    try:
        # Load and evaluate model
        model, results = load_and_evaluate_model(model_path, config)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETED!")
        print("="*50)
        print("Model performance:")
        print(f"  Mean reward: {results['mean_reward']:.2f}")
        print(f"  Mean energy: {results['mean_energy']:.2f} kWh")
        print(f"  Mean comfort violations: {results['mean_comfort_violation']:.2f} °C⋅hours")
        
        return model, results
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block.
    
    You can modify this section to:
    - Run complete training pipeline
    - Evaluate an existing model
    - Customize training parameters
    """
    
    print("Welcome to the PPO Training and Evaluation Guide!")
    print("=" * 60)
    print()
    print("This script demonstrates how to:")
    print("1. Train a PPO agent for building HVAC control")
    print("2. Evaluate the trained agent")
    print("3. Compare performance with random policy")
    print("4. Generate comprehensive reports")
    print()
    
    # Choose what to run
    mode = input("Choose mode:\n1. Run complete training pipeline\n2. Evaluate existing model\nEnter choice (1 or 2): ")
    
    if mode == "1":
        # Run complete training and evaluation
        run_complete_training_pipeline()
        
    elif mode == "2":
        # Evaluate existing model
        model_path = input("Enter path to saved model (without .zip): ")
        run_evaluation_only(model_path)
        
    else:
        print("Invalid choice. Running complete training pipeline by default.")
        run_complete_training_pipeline()