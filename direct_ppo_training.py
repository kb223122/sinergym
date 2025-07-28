#!/usr/bin/env python3
"""
Direct PPO Agent Training for Sinergym - Complete Implementation
================================================================

This script provides a complete PPO (Proximal Policy Optimization) training implementation
for Sinergym building control environments. Everything is self-contained with no external
config files required.

WHAT IS PPO?
============
PPO (Proximal Policy Optimization) is a Deep Reinforcement Learning algorithm that:
- Learns to control building HVAC systems through trial and error
- Balances energy efficiency with occupant comfort
- Uses a neural network to decide heating/cooling setpoints
- Gets rewards for good performance and penalties for poor performance

HOW PPO TRAINING WORKS:
=======================
1. OBSERVATION: AI observes building state (temperature, weather, energy use, etc.)
2. ACTION: AI sets heating and cooling setpoints based on current state
3. REWARD: AI gets positive reward for good energy/comfort balance, negative for bad
4. LEARNING: AI adjusts its decision-making to get better rewards over time

TRAINING PROCESS:
=================
1. Environment Setup: Create building simulation with wrappers for better training
2. Model Creation: Initialize PPO neural network with hyperparameters
3. Experience Collection: AI interacts with building, collects experience
4. Learning Updates: AI uses experience to improve its policy
5. Evaluation: Periodic testing to monitor progress
6. Model Saving: Save best performing model

Author: AI Assistant
Date: 2024
"""

# ============================================================================
# IMPORTS - All necessary libraries
# ============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Gymnasium - Standard RL environment interface
import gymnasium as gym

# Sinergym - Building simulation environment
import sinergym

# Stable Baselines 3 - PPO algorithm implementation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Sinergym utilities for wrappers and callbacks
from sinergym.utils.wrappers import (
    NormalizeObservation,    # Scales observations to help training
    NormalizeAction,         # Scales actions to help training
    LoggerWrapper,           # Records environment interactions
    CSVLogger               # Saves data to CSV files
)
from sinergym.utils.callbacks import LoggerEvalCallback

print("✓ All libraries imported successfully!")

# ============================================================================
# CONFIGURATION CLASS - All training settings in one place
# ============================================================================

class PPOTrainingConfig:
    """
    Complete configuration for PPO training.
    All settings are contained here - no external files needed.
    """
    
    def __init__(self):
        # ====================================================================
        # ENVIRONMENT SETTINGS
        # ====================================================================
        self.env_name = 'Eplus-5zone-hot-continuous-v1'  # Building simulation to use
        self.experiment_name = f"PPO_Direct_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Environment parameters (customize building behavior)
        self.env_params = {
            'timesteps_per_hour': 4,  # 15-minute timesteps (faster simulation)
            'runperiod': [1, 1, 1991, 31, 12, 1991],  # Full year simulation
            'reward': {
                'temperature_variables': ['air_temperature'],
                'energy_variables': ['HVAC_electricity_demand_rate'],
                'range_comfort_winter': [20.0, 23.5],  # Comfort range in winter (°C)
                'range_comfort_summer': [23.0, 26.0],  # Comfort range in summer (°C)
                'summer_start': [6, 1],  # June 1st
                'summer_final': [9, 30],  # September 30th
                'energy_weight': 0.5,  # Balance between energy and comfort
                'lambda_energy': 0.0001,  # Energy penalty coefficient
                'lambda_temperature': 1.0  # Comfort penalty coefficient
            }
        }
        
        # ====================================================================
        # TRAINING PARAMETERS
        # ====================================================================
        self.total_timesteps = 200000    # Total training steps (more = better learning)
        self.eval_freq = 10000           # Test every 10,000 steps
        self.n_eval_episodes = 3         # Episodes per evaluation
        self.final_eval_episodes = 5     # Episodes for final evaluation
        
        # ====================================================================
        # PPO HYPERPARAMETERS - The "knobs" that control learning
        # ====================================================================
        
        # Learning rate: How fast the AI learns
        # Too high = unstable training, too low = slow learning
        self.learning_rate = 3e-4  # 0.0003 (good default)
        
        # Experience collection: How much data to collect before learning
        self.n_steps = 2048  # Steps before each policy update
        
        # Training batches: How much data to process at once
        self.batch_size = 64  # Size of training batches
        
        # Training epochs: How many times to reuse each batch
        self.n_epochs = 10  # Epochs per update
        
        # Discount factor: How much future rewards matter
        self.gamma = 0.99  # 0-1, higher = more future-focused
        
        # Advantage estimation: Smoothing parameter for learning signal
        self.gae_lambda = 0.95  # 0-1, higher = more smoothing
        
        # PPO clipping: Prevents too large policy updates
        self.clip_range = 0.2  # Clipping range for policy updates
        
        # Entropy coefficient: Encourages exploration
        self.ent_coef = 0.01  # Higher = more random actions
        
        # Value function coefficient: How much to learn value estimates
        self.vf_coef = 0.5  # Value function loss weight
        
        # Maximum gradient norm: Prevents exploding gradients
        self.max_grad_norm = 0.5  # Gradient clipping
        
        # ====================================================================
        # OUTPUT SETTINGS
        # ====================================================================
        self.model_save_path = "./models/"
        self.logs_path = "./logs/"
        self.plots_path = "./plots/"
        
        # Create output directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
    
    def print_config(self):
        """Print current configuration for reference."""
        print("\n" + "="*60)
        print("PPO TRAINING CONFIGURATION")
        print("="*60)
        print(f"Environment: {self.env_name}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Steps per update: {self.n_steps}")
        print(f"Evaluation frequency: {self.eval_freq:,} steps")
        print(f"Model save path: {self.model_save_path}")
        print("="*60)

# ============================================================================
# ENVIRONMENT CREATION FUNCTIONS
# ============================================================================

def create_training_environment(config: PPOTrainingConfig) -> gym.Env:
    """
    Create and configure the training environment with all necessary wrappers.
    
    Wrappers are "filters" that modify the environment to help training:
    - NormalizeObservation: Scales sensor readings to similar ranges
    - NormalizeAction: Scales actions to [-1, 1] range for PPO
    - LoggerWrapper: Records all interactions for analysis
    - CSVLogger: Saves data to CSV files
    
    Args:
        config: Training configuration object
        
    Returns:
        gym.Env: Wrapped environment ready for training
    """
    print(f"\n🔧 Creating training environment: {config.env_name}")
    
    # Step 1: Create base environment with custom parameters
    env_params = config.env_params.copy()
    env_params['env_name'] = f"{config.experiment_name}_TRAIN"
    
    env = gym.make(config.env_name, **env_params)
    
    print(f"✓ Base environment created:")
    print(f"  - Name: {env.name}")
    print(f"  - Observations: {env.observation_space.shape[0]} variables")
    print(f"  - Actions: {env.action_space.shape[0]} variables (heating & cooling setpoints)")
    print(f"  - Episode length: {env.timestep_per_episode} timesteps")
    print(f"  - Step size: {env.step_size} seconds")
    
    # Step 2: Apply normalization wrappers
    print("  Applying normalization wrappers...")
    
    # Normalize observations: converts sensor readings to standard scale
    # This helps because AI works better when all inputs are in similar ranges
    # Example: temperature (20-30°C) vs energy (0-50000W) → both become ~0-1
    env = NormalizeObservation(env)
    print("    ✓ NormalizeObservation: scales sensor readings to similar ranges")
    
    # Normalize actions: converts actions to range [-1, 1]
    # PPO works best when actions are in this standard range
    # Example: setpoints (15-30°C) → (-1, 1)
    env = NormalizeAction(env)
    print("    ✓ NormalizeAction: scales actions to [-1, 1] range")
    
    # Step 3: Apply logging wrappers
    print("  Applying logging wrappers...")
    
    # LoggerWrapper records all environment interactions
    # Tracks: observations, actions, rewards, environment info
    env = LoggerWrapper(env)
    print("    ✓ LoggerWrapper: records all interactions")
    
    # CSVLogger saves interaction data to CSV files
    # Useful for plotting and analyzing performance later
    env = CSVLogger(env)
    print("    ✓ CSVLogger: saves data to CSV files")
    
    print(f"✓ Training environment ready!")
    print(f"  - Final observation space: {env.observation_space}")
    print(f"  - Final action space: {env.action_space}")
    
    return env

def create_evaluation_environment(config: PPOTrainingConfig, 
                                mean: Optional[np.ndarray] = None, 
                                var: Optional[np.ndarray] = None) -> gym.Env:
    """
    Create evaluation environment for testing the trained agent.
    
    This is similar to training environment but:
    - Uses fixed normalization parameters (from training)
    - Disables learning-related updates
    - Used only for testing, not training
    
    Args:
        config: Training configuration
        mean: Mean values for observation normalization (from training)
        var: Variance values for observation normalization (from training)
        
    Returns:
        gym.Env: Environment configured for evaluation
    """
    print(f"\n🔧 Creating evaluation environment: {config.env_name}")
    
    # Create base environment
    env_params = config.env_params.copy()
    env_params['env_name'] = f"{config.experiment_name}_EVAL"
    
    env = gym.make(config.env_name, **env_params)
    
    # Apply same wrappers as training
    print("  Applying wrappers (same as training)...")
    
    # Use fixed normalization parameters if provided (for fair evaluation)
    if mean is not None and var is not None:
        env = NormalizeObservation(env, mean=mean, var=var, automatic_update=False)
        print("    ✓ NormalizeObservation: using fixed parameters from training")
    else:
        env = NormalizeObservation(env)
        print("    ✓ NormalizeObservation: using dynamic parameters")
    
    env = NormalizeAction(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)
    
    print(f"✓ Evaluation environment ready!")
    return env

# ============================================================================
# PPO MODEL CREATION
# ============================================================================

def create_ppo_model(env: gym.Env, config: PPOTrainingConfig) -> PPO:
    """
    Create a PPO model with optimized hyperparameters for building control.
    
    PPO (Proximal Policy Optimization) is a Deep RL algorithm that:
    - Learns a policy (neural network) to map observations to actions
    - Uses "proximal" updates to prevent too large policy changes
    - Balances exploration (trying new actions) with exploitation (using known good actions)
    - Estimates value function to understand long-term rewards
    
    Args:
        env: Training environment
        config: Training configuration
        
    Returns:
        PPO: Configured PPO model ready for training
    """
    print(f"\n🤖 Creating PPO model...")
    
    # Create PPO model with all hyperparameters
    model = PPO(
        policy='MlpPolicy',              # Use Multi-Layer Perceptron (neural network)
        env=env,                         # Environment to learn in
        learning_rate=config.learning_rate,  # How fast to learn
        n_steps=config.n_steps,          # Steps before each update
        batch_size=config.batch_size,    # Training batch size
        n_epochs=config.n_epochs,        # Epochs per update
        gamma=config.gamma,               # Discount factor
        gae_lambda=config.gae_lambda,    # Advantage estimation smoothing
        clip_range=config.clip_range,    # PPO clipping parameter
        ent_coef=config.ent_coef,        # Entropy coefficient (exploration)
        vf_coef=config.vf_coef,          # Value function coefficient
        max_grad_norm=config.max_grad_norm,  # Gradient clipping
        verbose=1,                       # Print training progress
        seed=42                          # For reproducibility
    )
    
    print(f"✓ PPO model created successfully!")
    print(f"  - Policy: Neural network (MlpPolicy)")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Steps per update: {config.n_steps}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Epochs per update: {config.n_epochs}")
    print(f"  - Device: {model.device}")
    print(f"  - Seed: 42 (for reproducibility)")
    
    return model

# ============================================================================
# TRAINING CALLBACKS SETUP
# ============================================================================

def setup_training_callbacks(config: PPOTrainingConfig, 
                           eval_env: gym.Env) -> CallbackList:
    """
    Set up callbacks for monitoring training progress.
    
    Callbacks are functions that run during training to:
    - Evaluate model performance periodically
    - Save the best model
    - Log training metrics
    - Monitor for issues
    
    Args:
        config: Training configuration
        eval_env: Evaluation environment
        
    Returns:
        CallbackList: List of callbacks for training
    """
    print(f"\n📊 Setting up training callbacks...")
    
    callbacks = []
    
    # Evaluation callback: tests model periodically during training
    eval_callback = EvalCallback(
        eval_env=eval_env,                           # Environment for testing
        best_model_save_path=config.model_save_path, # Where to save best model
        log_path=config.logs_path,                   # Where to save logs
        eval_freq=config.eval_freq,                  # Test every N steps
        n_eval_episodes=config.n_eval_episodes,     # Episodes per evaluation
        deterministic=True,                          # Use best actions (no randomness)
        render=False,                               # Don't show visualization
        verbose=1                                   # Print evaluation results
    )
    callbacks.append(eval_callback)
    
    print(f"✓ Callbacks configured:")
    print(f"  - Evaluation frequency: every {config.eval_freq:,} steps")
    print(f"  - Episodes per evaluation: {config.n_eval_episodes}")
    print(f"  - Best model save path: {config.model_save_path}")
    print(f"  - Logs save path: {config.logs_path}")
    
    return CallbackList(callbacks)

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_ppo_agent(config: PPOTrainingConfig) -> Tuple[PPO, gym.Env, gym.Env]:
    """
    Complete PPO training pipeline.
    
    This function:
    1. Creates training and evaluation environments
    2. Creates PPO model with optimized hyperparameters
    3. Sets up monitoring callbacks
    4. Trains the model for specified timesteps
    5. Saves the final model
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple[PPO, gym.Env, gym.Env]: Trained model, training env, eval env
    """
    print("\n" + "="*80)
    print("🚀 STARTING PPO TRAINING PIPELINE")
    print("="*80)
    
    try:
        # Step 1: Create environments
        print("\n📦 PHASE 1: ENVIRONMENT SETUP")
        train_env = create_training_environment(config)
        eval_env = create_evaluation_environment(config)
        
        # Step 2: Create PPO model
        print("\n🤖 PHASE 2: MODEL CREATION")
        model = create_ppo_model(train_env, config)
        
        # Step 3: Set up training callbacks
        print("\n📊 PHASE 3: MONITORING SETUP")
        callbacks = setup_training_callbacks(config, eval_env)
        
        # Step 4: Start training
        print("\n🎯 PHASE 4: TRAINING EXECUTION")
        print(f"Training for {config.total_timesteps:,} timesteps...")
        print("This will take several minutes to hours depending on your computer.")
        print()
        print("What you'll see during training:")
        print("- 'rollout/ep_rew_mean': Average reward per episode (higher is better)")
        print("- 'train/learning_rate': Current learning rate")
        print("- 'eval/mean_reward': Test performance (how well AI is doing)")
        print("- 'time/fps': Training speed (steps per second)")
        print()
        
        # This is where the actual learning happens!
        # The AI will:
        # 1. Observe building state (temperature, weather, energy use, etc.)
        # 2. Choose actions (heating and cooling setpoints)
        # 3. Get rewards based on energy efficiency and comfort
        # 4. Learn from this experience to make better choices next time
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True    # Show progress bar
        )
        
        # Step 5: Save final model
        print(f"\n💾 PHASE 5: MODEL SAVING")
        final_model_path = os.path.join(config.model_save_path, "final_model")
        model.save(final_model_path)
        print(f"✓ Final model saved to: {final_model_path}.zip")
        
        # Clean up environments
        train_env.close()
        eval_env.close()
        
        print(f"\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return model, train_env, eval_env
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save model and close environments
        try:
            if 'model' in locals():
                emergency_path = os.path.join(config.model_save_path, "emergency_model")
                model.save(emergency_path)
                print(f"💾 Emergency model save to: {emergency_path}.zip")
        except:
            pass
        
        try:
            if 'train_env' in locals():
                train_env.close()
            if 'eval_env' in locals():
                eval_env.close()
        except:
            pass
        
        raise e

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_trained_model(model: PPO, 
                         eval_env: gym.Env, 
                         config: PPOTrainingConfig,
                         num_episodes: int = 5) -> Dict[str, Any]:
    """
    Evaluate a trained PPO model over multiple episodes.
    
    This function:
    1. Runs the trained model on evaluation episodes
    2. Collects performance metrics (rewards, energy, comfort)
    3. Calculates summary statistics
    4. Compares with random policy baseline
    
    Args:
        model: Trained PPO model
        eval_env: Evaluation environment
        config: Training configuration
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    print(f"\n" + "="*60)
    print("🧪 EVALUATING TRAINED MODEL")
    print("="*60)
    
    print(f"Running {num_episodes} evaluation episodes...")
    
    # Initialize result tracking
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\n📊 Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Reset environment for new episode
        obs, info = eval_env.reset()
        
        # Initialize episode tracking
        episode_reward = 0              # Total reward for this episode
        episode_energy = 0              # Total energy consumption
        episode_comfort_violation = 0   # Total comfort violations
        episode_length = 0              # Number of steps in episode
        
        # Run one complete episode
        terminated = truncated = False
        # terminated: episode ended naturally (simulation complete)
        # truncated: episode ended early (time limit, error, etc.)
        
        while not (terminated or truncated):
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            # deterministic=True: use best action (no exploration)
            # action contains [heating_setpoint, cooling_setpoint]
            
            # Take action in environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            # obs: new observation after taking action
            # reward: immediate reward for this action
            # terminated/truncated: episode ending flags
            # info: additional environment information
            
            # Update episode tracking
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            episode_length += 1
            
            # Print progress every 1000 steps
            if episode_length % 1000 == 0:
                print(f"  Step {episode_length}: Reward = {reward:.2f}, "
                      f"Energy = {info.get('total_power_demand', 0):.1f}W")
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violation)
        episode_lengths.append(episode_length)
        
        # Print episode summary
        print(f"Episode {episode + 1} completed:")
        print(f"  Steps: {episode_length}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Total energy: {episode_energy:.2f} kWh")
        print(f"  Comfort violations: {episode_comfort_violation:.2f} °C⋅hours")
    
    # Calculate summary statistics
    results = {
        'episode_rewards': episode_rewards,
        'episode_energies': episode_energies,
        'episode_comfort_violations': episode_comfort_violations,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'std_energy': np.std(episode_energies),
        'mean_comfort_violation': np.mean(episode_comfort_violations),
        'std_comfort_violation': np.std(episode_comfort_violations),
        'mean_episode_length': np.mean(episode_lengths)
    }
    
    # Print summary
    print(f"\n" + "="*40)
    print("📈 EVALUATION SUMMARY")
    print("="*40)
    print(f"Episodes evaluated: {num_episodes}")
    print(f"Mean episode reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean energy consumption: {results['mean_energy']:.2f} ± {results['std_energy']:.2f} kWh")
    print(f"Mean comfort violations: {results['mean_comfort_violation']:.2f} ± {results['std_comfort_violation']:.2f} °C⋅hours")
    print(f"Mean episode length: {results['mean_episode_length']:.0f} steps")
    
    return results

def compare_with_random_policy(eval_env: gym.Env, 
                             config: PPOTrainingConfig,
                             num_episodes: int = 3) -> Dict[str, Any]:
    """
    Compare trained model with random policy (baseline).
    
    This shows how much better the trained AI is compared to
    completely random control actions.
    
    Args:
        eval_env: Evaluation environment
        config: Training configuration
        num_episodes: Number of episodes to test
        
    Returns:
        Dict[str, Any]: Random policy results
    """
    print(f"\n" + "="*60)
    print("🎲 COMPARING WITH RANDOM POLICY")
    print("="*60)
    
    print(f"Running {num_episodes} episodes with random actions...")
    print("(This shows how bad completely random control would be)")
    
    # Track results
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    
    for episode in range(num_episodes):
        print(f"\n🎲 Random Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, info = eval_env.reset()
        
        # Track metrics
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        step_count = 0
        
        # Run episode with random actions
        terminated = truncated = False
        while not (terminated or truncated):
            # Take completely random action (no intelligence!)
            action = eval_env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Track what happened
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            step_count += 1
        
        # Store results
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violation)
        
        print(f"  Random episode reward: {episode_reward:.2f}")
    
    # Calculate summary
    results = {
        'mean_reward': np.mean(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'mean_comfort_violation': np.mean(episode_comfort_violations)
    }
    
    print(f"\n🎲 Random Policy Summary:")
    print(f"Average reward: {results['mean_reward']:.2f}")
    print(f"Average energy: {results['mean_energy']:.2f} kWh")
    print(f"Average comfort violations: {results['mean_comfort_violation']:.2f} °C⋅hours")
    
    return results

# ============================================================================
# MODEL LOADING AND TESTING
# ============================================================================

def load_and_evaluate_model(model_path: str, 
                          config: PPOTrainingConfig) -> Tuple[PPO, Dict[str, Any]]:
    """
    Load a previously saved model and evaluate it.
    
    Args:
        model_path: Path to saved model (without .zip extension)
        config: Training configuration
        
    Returns:
        Tuple[PPO, Dict[str, Any]]: Loaded model and evaluation results
    """
    print(f"\n" + "="*60)
    print("📂 LOADING SAVED MODEL")
    print("="*60)
    
    print(f"Loading model from: {model_path}")
    
    # Load the model
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure the model file exists and the path is correct.")
        return None, None
    
    # Create evaluation environment
    eval_env = create_evaluation_environment(config)
    
    # Test the loaded model
    print("Testing loaded model...")
    results = evaluate_trained_model(model, eval_env, config, num_episodes=3)
    
    eval_env.close()
    return model, results

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_complete_training_pipeline():
    """
    Run the complete PPO training pipeline from start to finish.
    
    This function will:
    1. Create configuration
    2. Train a PPO model
    3. Evaluate the trained model
    4. Compare with random policy
    5. Show comprehensive results
    """
    print("="*80)
    print("🎯 COMPLETE PPO TRAINING PIPELINE")
    print("="*80)
    print()
    print("This pipeline will:")
    print("1. Train an AI to control building HVAC systems")
    print("2. Test how well the AI learned")
    print("3. Compare the AI with random actions")
    print("4. Show comprehensive performance analysis")
    print()
    print("The AI will learn to balance:")
    print("- Energy efficiency (using less electricity)")
    print("- Occupant comfort (keeping good temperatures)")
    print()
    input("Press Enter to start...")
    
    try:
        # Step 1: Create configuration
        print("\n⚙️  STEP 1: CONFIGURATION")
        config = PPOTrainingConfig()
        config.print_config()
        
        # Step 2: Train the model
        print("\n🎯 STEP 2: TRAINING THE AI")
        trained_model, train_env, eval_env = train_ppo_agent(config)
        
        # Step 3: Evaluate the trained model
        print("\n🧪 STEP 3: TESTING THE TRAINED AI")
        trained_results = evaluate_trained_model(trained_model, eval_env, config, 
                                              num_episodes=config.final_eval_episodes)
        
        # Step 4: Compare with random policy
        print("\n🎲 STEP 4: COMPARING WITH RANDOM ACTIONS")
        random_results = compare_with_random_policy(eval_env, config, num_episodes=3)
        
        # Step 5: Show final comparison
        print("\n📊 FINAL RESULTS COMPARISON")
        print("="*80)
        print(f"{'Metric':<30} {'Trained AI':<20} {'Random Policy':<20} {'AI Better?'}")
        print("-" * 80)
        
        # Compare rewards (higher is better)
        reward_better = "✓ YES" if trained_results['mean_reward'] > random_results['mean_reward'] else "✗ NO"
        print(f"{'Average Reward':<30} {trained_results['mean_reward']:<20.2f} {random_results['mean_reward']:<20.2f} {reward_better}")
        
        # Compare energy (lower is better)
        energy_improvement = ((random_results['mean_energy'] - trained_results['mean_energy']) / random_results['mean_energy']) * 100
        energy_better = "✓ YES" if trained_results['mean_energy'] < random_results['mean_energy'] else "✗ NO"
        print(f"{'Energy Use (kWh)':<30} {trained_results['mean_energy']:<20.2f} {random_results['mean_energy']:<20.2f} {energy_better}")
        
        # Compare comfort (lower is better)
        comfort_improvement = ((random_results['mean_comfort_violation'] - trained_results['mean_comfort_violation']) / random_results['mean_comfort_violation']) * 100
        comfort_better = "✓ YES" if trained_results['mean_comfort_violation'] < random_results['mean_comfort_violation'] else "✗ NO"
        print(f"{'Comfort Violations':<30} {trained_results['mean_comfort_violation']:<20.2f} {random_results['mean_comfort_violation']:<20.2f} {comfort_better}")
        
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n📈 Performance Summary:")
        print(f"✓ Energy improvement: {energy_improvement:+.1f}%")
        print(f"✓ Comfort improvement: {comfort_improvement:+.1f}%")
        print(f"✓ Reward improvement: {trained_results['mean_reward'] - random_results['mean_reward']:+.2f}")
        
        print(f"\n📁 Files created:")
        print(f"✓ Trained model: {config.model_save_path}final_model.zip")
        print(f"✓ Best model: {config.model_save_path}best_model.zip")
        print(f"✓ Training logs: {config.logs_path}")
        
        print(f"\n🚀 Next steps to improve your AI:")
        print("• Increase total_timesteps for longer training")
        print("• Adjust learning_rate (try 0.0001 or 0.001)")
        print("• Try different reward function weights")
        print("• Experiment with different environments")
        
        return trained_model, trained_results, random_results
        
    except Exception as e:
        print(f"\n❌ Error during pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_evaluation_only(model_path: str):
    """
    Run evaluation only on a previously trained model.
    
    Args:
        model_path: Path to saved model
    """
    print("="*60)
    print("🧪 EVALUATION ONLY MODE")
    print("="*60)
    
    config = PPOTrainingConfig()
    model, results = load_and_evaluate_model(model_path, config)
    
    if model is not None:
        print(f"\n✅ Evaluation completed!")
        print(f"Model performance: {results['mean_reward']:.2f} average reward")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🤖 Welcome to Direct PPO Training for Sinergym!")
    print()
    print("Choose your mode:")
    print("1. Complete training pipeline (recommended) - trains and evaluates")
    print("2. Evaluation only - test a previously trained model")
    print("3. Quick demo - minimal training for testing")
    print()
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n🎯 Starting complete training pipeline...")
        run_complete_training_pipeline()
        
    elif choice == "2":
        model_path = input("\nEnter path to saved model (without .zip): ").strip()
        run_evaluation_only(model_path)
        
    elif choice == "3":
        print("\n⚡ Starting quick demo...")
        # Modify config for quick demo
        config = PPOTrainingConfig()
        config.total_timesteps = 10000  # Very short training
        config.eval_freq = 5000
        print("Quick demo configuration:")
        config.print_config()
        
        # Run training with modified config
        trained_model, train_env, eval_env = train_ppo_agent(config)
        results = evaluate_trained_model(trained_model, eval_env, config, num_episodes=2)
        
    else:
        print("\n❓ Invalid choice. Starting complete pipeline...")
        run_complete_training_pipeline()
    
    print(f"\n🎯 Thanks for using Direct PPO Training!")
    print(f"You now have a working PPO agent for building control!")