#!/usr/bin/env python3
"""
Corrected Sinergym PPO Training with Episode-wise Training
==========================================================

This script fixes the parameter passing issue and implements episode-wise training
with parameter verification to confirm that changes are actually applied.

Key fixes:
1. Pass runperiod and timesteps_per_hour through config_params
2. Implement episode-wise training instead of timestep-wise
3. Add parameter verification at each step
4. Print runperiod and reward weights for confirmation
"""

import gymnasium as gym
import numpy as np
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sinergym.utils.wrappers import (
    NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger
)

def create_custom_env(env_name, config_params, reward_kwargs):
    """Create environment with custom parameters."""
    
    # Create environment with config_params
    env = gym.make(env_name, config_params=config_params, reward_kwargs=reward_kwargs)
    
    # Apply standard wrappers
    env = NormalizeObservation(env)
    env = NormalizeAction(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)
    
    return env

def print_env_info(env, step_info=None):
    """Print environment information and current step details."""
    print(f"\n{'='*80}")
    print(f"ENVIRONMENT INFORMATION")
    print(f"{'='*80}")
    print(f"Episode length: {env.get_wrapper_attr('timestep_per_episode')} timesteps")
    print(f"Runperiod: {env.get_wrapper_attr('runperiod')}")
    print(f"Observation space: {env.observation_space.shape[0]} variables")
    print(f"Action space: {env.action_space.shape[0]} variables")
    
    if step_info:
        print(f"\n{'='*80}")
        print(f"STEP INFORMATION")
        print(f"{'='*80}")
        print(f"Current timestep: {step_info.get('timestep', 'N/A')}")
        print(f"Current episode: {step_info.get('episode', 'N/A')}")
        print(f"Current reward: {step_info.get('reward', 'N/A'):.4f}")
        print(f"Temperature: {step_info.get('temperature', 'N/A'):.2f}°C")
        print(f"Energy consumption: {step_info.get('energy', 'N/A'):.2f} W")

def verify_parameters(env, expected_config, expected_reward):
    """Verify that the parameters were applied correctly."""
    print(f"\n{'='*80}")
    print(f"PARAMETER VERIFICATION")
    print(f"{'='*80}")
    
    # Get actual runperiod from environment
    actual_runperiod = env.get_wrapper_attr('runperiod')
    expected_runperiod = expected_config['runperiod']
    
    print(f"Expected runperiod: {expected_runperiod}")
    print(f"Actual runperiod: {actual_runperiod}")
    
    # Check if runperiod matches
    if actual_runperiod['start_month'] == expected_runperiod[1] and \
       actual_runperiod['start_day'] == expected_runperiod[0] and \
       actual_runperiod['end_month'] == expected_runperiod[4] and \
       actual_runperiod['end_day'] == expected_runperiod[3]:
        print("✅ Runperiod applied correctly!")
    else:
        print("❌ Runperiod mismatch!")
    
    # Get reward function parameters
    reward_func = env.unwrapped.reward
    print(f"\nReward function type: {type(reward_func).__name__}")
    
    if hasattr(reward_func, 'W_energy'):
        print(f"Energy weight: {reward_func.W_energy}")
        print(f"Expected energy weight: {expected_reward['energy_weight']}")
        if abs(reward_func.W_energy - expected_reward['energy_weight']) < 0.01:
            print("✅ Energy weight applied correctly!")
        else:
            print("❌ Energy weight mismatch!")
    
    if hasattr(reward_func, 'lambda_temp'):
        print(f"Temperature penalty: {reward_func.lambda_temp}")
        print(f"Expected temperature penalty: {expected_reward['lambda_temperature']}")
        if abs(reward_func.lambda_temp - expected_reward['lambda_temperature']) < 0.01:
            print("✅ Temperature penalty applied correctly!")
        else:
            print("❌ Temperature penalty mismatch!")

def train_ppo_episode_wise(env, num_episodes=5, name="Episode-wise Training"):
    """Train PPO agent episode-wise instead of timestep-wise."""
    print(f"\n{'='*80}")
    print(f"STARTING EPISODE-WISE TRAINING: {name}")
    print(f"{'='*80}")
    
    # Create evaluation environment
    eval_env = create_custom_env(
        env.unwrapped.spec.id, 
        env.unwrapped.config, 
        env.unwrapped.reward_kwargs
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{name}/",
        log_path=f"./logs/{name}/",
        eval_freq=1,  # Evaluate every episode
        n_eval_episodes=1,
        deterministic=True,
        verbose=1
    )
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )
    
    # Calculate total timesteps for the specified number of episodes
    timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
    total_timesteps = num_episodes * timesteps_per_episode
    
    print(f"Training for {num_episodes} episodes")
    print(f"Timesteps per episode: {timesteps_per_episode}")
    print(f"Total timesteps: {total_timesteps}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    eval_env.close()
    return model

def test_environment_step_by_step(env, num_steps=10):
    """Test environment step by step to verify parameters."""
    print(f"\n{'='*80}")
    print(f"STEP-BY-STEP ENVIRONMENT TEST")
    print(f"{'='*80}")
    
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(num_steps):
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        
        # Extract relevant information
        step_info = {
            'timestep': step + 1,
            'episode': 1,
            'reward': reward,
            'temperature': obs[8] if len(obs) > 8 else 'N/A',  # air_temperature index
            'energy': obs[-1] if len(obs) > 0 else 'N/A'       # energy index
        }
        
        print_env_info(env, step_info)
        
        if terminated or truncated:
            print(f"\nEpisode ended after {step + 1} steps")
            break
    
    print(f"\nTotal episode reward: {episode_reward:.4f}")
    env.close()

def main():
    """Main function with corrected parameter passing."""
    
    # Custom environment configuration
    config_params = {
        # Time period: Summer months only
        'runperiod': (1, 6, 1991, 31, 8, 1991),  # June 1 to August 31, 1991
        
        # Simulation resolution
        'timesteps_per_hour': 4
    }
    
    # Custom reward function parameters
    reward_kwargs = {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': (20.0, 23.5),
        'range_comfort_summer': (23.0, 26.0),
        'summer_start': (6, 1),
        'summer_final': (8, 31),
        'energy_weight': 0.7,        # Focus on energy efficiency
        'lambda_energy': 0.0001,
        'lambda_temperature': 0.8     # Moderate comfort penalty
    }
    
    print("Creating environment with custom parameters...")
    
    # Create environment with correct parameter passing
    env = create_custom_env(
        'Eplus-5zone-hot-continuous-v1', 
        config_params, 
        reward_kwargs
    )
    
    # Print initial environment information
    print_env_info(env)
    
    # Verify parameters were applied correctly
    verify_parameters(env, config_params, reward_kwargs)
    
    # Test environment step by step
    test_environment_step_by_step(env, num_steps=5)
    
    # Train PPO agent episode-wise
    model = train_ppo_episode_wise(env, num_episodes=3, name="summer_energy_focused")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    env.close()

if __name__ == "__main__":
    main()