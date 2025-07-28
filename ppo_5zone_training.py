#!/usr/bin/env python3
"""
PPO Training with Custom Reward Weights and Run Periods
=======================================================

Complete PPO training example for Eplus-5zone-hot-continuous-v1 environment
with customizable reward weights and run periods.

Features:
- Custom reward function with adjustable weights
- Configurable run periods (timestep size, episode duration)
- PPO agent training with monitoring
- Environment verification and testing
- Training progress tracking

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

# =============================================================================
# STEP 1: IMPORTS AND SETUP
# =============================================================================

print("🚀 PPO Training with Custom 5Zone Environment")
print("=" * 60)

try:
    import gymnasium as gym
    import sinergym
    from sinergym.utils.rewards import LinearReward
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install sinergym[drl] stable-baselines3 gymnasium")
    sys.exit(1)

# =============================================================================
# STEP 2: CUSTOM REWARD CONFIGURATION
# =============================================================================

def create_custom_reward_function(energy_weight=0.5, lambda_energy=1e-4, lambda_temperature=1.0):
    """
    Create a custom reward function with specified weights.
    
    Args:
        energy_weight (float): Weight for energy term (0.0 to 1.0)
        lambda_energy (float): Scaling factor for energy penalty
        lambda_temperature (float): Scaling factor for temperature penalty
    
    Returns:
        LinearReward: Custom reward function
    """
    
    return LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(20.0, 23.5),
        range_comfort_summer=(23.0, 26.0),
        energy_weight=energy_weight,
        lambda_energy=lambda_energy,
        lambda_temperature=lambda_temperature
    )

def print_reward_config(energy_weight, lambda_energy, lambda_temperature):
    """Print reward configuration details."""
    print(f"\n📊 CUSTOM REWARD CONFIGURATION")
    print("-" * 50)
    print(f"Energy Weight: {energy_weight:.2f} ({energy_weight*100:.0f}%)")
    print(f"Lambda Energy: {lambda_energy:.2e}")
    print(f"Lambda Temperature: {lambda_temperature:.2f}")
    print(f"Comfort Weight: {1-energy_weight:.2f} ({(1-energy_weight)*100:.0f}%)")
    print(f"Reward Formula: R = -{energy_weight:.2f} × {lambda_energy:.2e} × energy - {1-energy_weight:.2f} × {lambda_temperature:.2f} × temp_violation")

# =============================================================================
# STEP 3: RUN PERIOD CONFIGURATION
# =============================================================================

def create_run_period_config(timestep_per_hour=1, start_month=1, start_day=1, end_month=12, end_day=31):
    """
    Create run period configuration.
    
    Args:
        timestep_per_hour (int): Number of timesteps per hour (1, 2, 4, etc.)
        start_month (int): Starting month (1-12)
        start_day (int): Starting day (1-31)
        end_month (int): Ending month (1-12)
        end_day (int): Ending day (1-31)
    
    Returns:
        dict: Run period configuration
    """
    
    runperiod = (start_month, start_day, end_month, end_day)
    
    # Calculate timesteps per episode based on run period
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if start_month == end_month:
        total_days = end_day - start_day + 1
    else:
        total_days = (days_per_month[start_month-1] - start_day + 1)
        for month in range(start_month + 1, end_month):
            total_days += days_per_month[month-1]
        total_days += end_day
    
    timesteps_per_episode = total_days * 24 // timestep_per_hour
    
    config = {
        'timestep_per_hour': timestep_per_hour,
        'runperiod': runperiod,
        'timesteps_per_episode': timesteps_per_episode,
        'total_days': total_days,
        'total_hours': total_days * 24
    }
    
    return config

def print_run_period_config(config):
    """Print run period configuration details."""
    print(f"\n⏱️ RUN PERIOD CONFIGURATION")
    print("-" * 50)
    print(f"Timesteps per Hour: {config['timestep_per_hour']}")
    print(f"Run Period: {config['runperiod']} (Month/Day to Month/Day)")
    print(f"Total Days: {config['total_days']}")
    print(f"Total Hours: {config['total_hours']}")
    print(f"Timesteps per Episode: {config['timesteps_per_episode']:,}")
    
    # Calculate episode duration
    hours_per_episode = config['timesteps_per_episode'] / config['timestep_per_hour']
    days_per_episode = hours_per_episode / 24
    print(f"Episode Duration: {hours_per_episode:.0f} hours ({days_per_episode:.1f} days)")

# =============================================================================
# STEP 4: ENVIRONMENT CREATION AND CONFIGURATION
# =============================================================================

def create_custom_environment(reward_config=None, run_period_config=None):
    """
    Create and configure the 5Zone environment with custom settings.
    
    Args:
        reward_config (dict): Reward function configuration
        run_period_config (dict): Run period configuration
    
    Returns:
        gym.Env: Configured environment
    """
    
    print(f"\n🔧 CREATING CUSTOM ENVIRONMENT")
    print("=" * 50)
    
    try:
        # Create base environment
        env = gym.make('Eplus-5zone-hot-continuous-v1')
        
        # Apply custom reward function if provided
        if reward_config:
            print(f"📊 Applying custom reward configuration:")
            print(f"   Energy Weight: {reward_config['energy_weight']:.2f}")
            print(f"   Lambda Energy: {reward_config['lambda_energy']:.2e}")
            print(f"   Lambda Temperature: {reward_config['lambda_temperature']:.2f}")
            
            custom_reward = create_custom_reward_function(
                energy_weight=reward_config['energy_weight'],
                lambda_energy=reward_config['lambda_energy'],
                lambda_temperature=reward_config['lambda_temperature']
            )
            env.reward_fn = custom_reward
        
        # Apply custom run period if provided
        if run_period_config:
            print(f"⏱️ Applying custom run period configuration:")
            print(f"   Timesteps per Hour: {run_period_config['timestep_per_hour']}")
            print(f"   Run Period: {run_period_config['runperiod']}")
            print(f"   Timesteps per Episode: {run_period_config['timesteps_per_episode']:,}")
            
            env.timestep_per_hour = run_period_config['timestep_per_hour']
            env.runperiod = run_period_config['runperiod']
        
        print("✅ Environment created successfully")
        return env
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return None

def verify_environment_configuration(env, reward_config=None, run_period_config=None):
    """Verify that environment has correct configurations."""
    
    print(f"\n🔍 VERIFYING ENVIRONMENT CONFIGURATION")
    print("=" * 50)
    
    # Verify run period settings
    if run_period_config:
        print(f"Timestep per Hour: {env.timestep_per_hour} (Expected: {run_period_config['timestep_per_hour']})")
        print(f"Run Period: {env.runperiod} (Expected: {run_period_config['runperiod']})")
        print(f"Timesteps per Episode: {env.timestep_per_episode:,} (Expected: {run_period_config['timesteps_per_episode']:,})")
    
    # Verify reward function
    if reward_config and hasattr(env, 'reward_fn') and env.reward_fn is not None:
        print(f"Reward Function: {type(env.reward_fn).__name__}")
        
        # Check reward parameters
        if hasattr(env.reward_fn, 'energy_weight'):
            print(f"Energy Weight: {env.reward_fn.energy_weight:.2f} (Expected: {reward_config['energy_weight']:.2f})")
        if hasattr(env.reward_fn, 'lambda_energy'):
            print(f"Lambda Energy: {env.reward_fn.lambda_energy:.2e} (Expected: {reward_config['lambda_energy']:.2e})")
        if hasattr(env.reward_fn, 'lambda_temperature'):
            print(f"Lambda Temperature: {env.reward_fn.lambda_temperature:.2f} (Expected: {reward_config['lambda_temperature']:.2f})")
    else:
        print("Using default reward function")
    
    # Print environment info
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

def test_environment(env, num_steps=10):
    """Test the environment with random actions."""
    
    print(f"\n🧪 TESTING ENVIRONMENT ({num_steps} steps)")
    print("=" * 50)
    
    try:
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation range: [{obs.min():.2f}, {obs.max():.2f}]")
        
        total_reward = 0
        rewards = []
        
        for step in range(num_steps):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            
            if step < 3:  # Print first few steps
                print(f"Step {step+1}: Action = {action}, Reward = {reward:.4f}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step+1}")
                break
        
        print(f"\nTest Results:")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Average Reward: {np.mean(rewards):.4f}")
        print(f"  Reward Std: {np.std(rewards):.4f}")
        print(f"  Steps Completed: {len(rewards)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing environment: {e}")
        return False

# =============================================================================
# STEP 5: PPO TRAINING SETUP
# =============================================================================

def create_ppo_model(env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10):
    """
    Create a PPO model with custom hyperparameters.
    
    Args:
        env: Training environment
        learning_rate (float): Learning rate
        n_steps (int): Number of steps per update
        batch_size (int): Batch size
        n_epochs (int): Number of epochs per update
    
    Returns:
        PPO: Configured PPO model
    """
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./logs/",
        verbose=1
    )
    
    return model

def setup_callbacks(env, eval_env, save_freq=10000, eval_freq=5000):
    """
    Setup training callbacks.
    
    Args:
        env: Training environment
        eval_env: Evaluation environment
        save_freq (int): Frequency for saving checkpoints
        eval_freq (int): Frequency for evaluation
    
    Returns:
        list: List of callbacks
    """
    
    # Create logs directory
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models/",
        name_prefix="ppo_5zone"
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    return [checkpoint_callback, eval_callback]

# =============================================================================
# STEP 6: TRAINING FUNCTION
# =============================================================================

def train_ppo_agent(env, eval_env, total_timesteps=100000, learning_rate=3e-4):
    """
    Train a PPO agent with the given environment.
    
    Args:
        env: Training environment
        eval_env: Evaluation environment
        total_timesteps (int): Total training timesteps
        learning_rate (float): Learning rate
    
    Returns:
        PPO: Trained model
    """
    
    print(f"\n🎯 STARTING PPO TRAINING")
    print("=" * 50)
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Timesteps per Episode: {env.timestep_per_episode:,}")
    print(f"Estimated Episodes: {total_timesteps // env.timestep_per_episode}")
    
    # Create model
    model = create_ppo_model(env, learning_rate=learning_rate)
    
    # Setup callbacks
    callbacks = setup_callbacks(env, eval_env)
    
    # Start training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        
        return model
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

# =============================================================================
# STEP 7: EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model, env, num_episodes=5):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained PPO model
        env: Evaluation environment
        num_episodes (int): Number of episodes to evaluate
    
    Returns:
        dict: Evaluation results
    """
    
    print(f"\n📊 EVALUATING MODEL ({num_episodes} episodes)")
    print("=" * 50)
    
    episode_rewards = []
    episode_lengths = []
    energy_consumptions = []
    comfort_violations = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_energy = 0
        episode_comfort_violation = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Collect metrics if available
            if 'total_power_demand' in info:
                episode_energy += info['total_power_demand']
            if 'total_temperature_violation' in info:
                episode_comfort_violation += info['total_temperature_violation']
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        energy_consumptions.append(episode_energy)
        comfort_violations.append(episode_comfort_violation)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.4f}, Length = {episode_length}")
    
    # Calculate statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_energy': np.mean(energy_consumptions) if energy_consumptions else None,
        'mean_comfort_violation': np.mean(comfort_violations) if comfort_violations else None
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"  Mean Episode Length: {results['mean_length']:.1f}")
    if results['mean_energy']:
        print(f"  Mean Energy Consumption: {results['mean_energy']:.2f}")
    if results['mean_comfort_violation']:
        print(f"  Mean Comfort Violation: {results['mean_comfort_violation']:.4f}")
    
    return results

# =============================================================================
# STEP 8: MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training script."""
    
    print("🚀 Starting PPO Training with Custom 5Zone Environment")
    print("=" * 80)
    
    # =============================================================================
    # CONFIGURATION SECTION - MODIFY THESE PARAMETERS AS NEEDED
    # =============================================================================
    
    # Reward configuration
    reward_config = {
        'energy_weight': 0.7,        # 70% weight on energy, 30% on comfort
        'lambda_energy': 2e-4,       # Higher energy penalty
        'lambda_temperature': 0.8     # Lower comfort penalty
    }
    
    # Run period configuration
    run_period_config = create_run_period_config(
        timestep_per_hour=2,          # 2-hour timesteps
        start_month=6,                # June
        start_day=1,                  # 1st
        end_month=8,                  # August
        end_day=31                    # 31st (Summer only)
    )
    
    # Training configuration
    training_config = {
        'total_timesteps': 50000,     # Total training timesteps
        'learning_rate': 3e-4,        # Learning rate
        'eval_episodes': 3            # Episodes for evaluation
    }
    
    # =============================================================================
    # PRINT CONFIGURATIONS
    # =============================================================================
    
    print_reward_config(
        reward_config['energy_weight'],
        reward_config['lambda_energy'],
        reward_config['lambda_temperature']
    )
    
    print_run_period_config(run_period_config)
    
    print(f"\n🎯 TRAINING CONFIGURATION")
    print("-" * 50)
    print(f"Total Timesteps: {training_config['total_timesteps']:,}")
    print(f"Learning Rate: {training_config['learning_rate']}")
    print(f"Evaluation Episodes: {training_config['eval_episodes']}")
    
    # =============================================================================
    # CREATE ENVIRONMENTS
    # =============================================================================
    
    # Create training environment
    train_env = create_custom_environment(reward_config, run_period_config)
    if train_env is None:
        print("❌ Failed to create training environment")
        return
    
    # Create evaluation environment (same configuration)
    eval_env = create_custom_environment(reward_config, run_period_config)
    if eval_env is None:
        print("❌ Failed to create evaluation environment")
        train_env.close()
        return
    
    # Verify configurations
    verify_environment_configuration(train_env, reward_config, run_period_config)
    
    # Test environments
    print(f"\n🧪 Testing Training Environment:")
    if not test_environment(train_env, num_steps=20):
        print("❌ Training environment test failed")
        train_env.close()
        eval_env.close()
        return
    
    print(f"\n🧪 Testing Evaluation Environment:")
    if not test_environment(eval_env, num_steps=10):
        print("❌ Evaluation environment test failed")
        train_env.close()
        eval_env.close()
        return
    
    # =============================================================================
    # TRAIN PPO AGENT
    # =============================================================================
    
    try:
        # Train the agent
        model = train_ppo_agent(
            train_env,
            eval_env,
            total_timesteps=training_config['total_timesteps'],
            learning_rate=training_config['learning_rate']
        )
        
        if model is None:
            print("❌ Training failed")
            train_env.close()
            eval_env.close()
            return
        
        # =============================================================================
        # EVALUATE TRAINED MODEL
        # =============================================================================
        
        # Evaluate the model
        results = evaluate_model(model, eval_env, num_episodes=training_config['eval_episodes'])
        
        # =============================================================================
        # SAVE MODEL
        # =============================================================================
        
        model_path = f"./models/ppo_5zone_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # =============================================================================
        # FINAL SUMMARY
        # =============================================================================
        
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Configuration Used:")
        print(f"  Energy Weight: {reward_config['energy_weight']:.2f}")
        print(f"  Lambda Energy: {reward_config['lambda_energy']:.2e}")
        print(f"  Lambda Temperature: {reward_config['lambda_temperature']:.2f}")
        print(f"  Run Period: {run_period_config['runperiod']}")
        print(f"  Timesteps per Hour: {run_period_config['timestep_per_hour']}")
        print(f"  Total Training Timesteps: {training_config['total_timesteps']:,}")
        print(f"  Final Mean Reward: {results['mean_reward']:.4f}")
        
        print(f"\n📁 Files Created:")
        print(f"  Model: {model_path}")
        print(f"  Logs: ./logs/")
        print(f"  Checkpoints: ./models/")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    main()