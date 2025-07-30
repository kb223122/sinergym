#!/usr/bin/env python3
"""
Sinergym Customization Examples
===============================

This script demonstrates how to customize runperiod and reward weights
in Sinergym environments for PPO training.

Features demonstrated:
1. Different runperiod configurations
2. Various reward weight customizations
3. Practical examples for different use cases
"""

import gymnasium as gym
import numpy as np
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sinergym.utils.wrappers import (
    NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger
)

def create_custom_env(env_name, env_params):
    """Create environment with custom parameters."""
    env = gym.make(env_name, **env_params)
    
    # Apply standard wrappers
    env = NormalizeObservation(env)
    env = NormalizeAction(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)
    
    return env

def print_env_info(env, name):
    """Print environment information."""
    print(f"\n{'='*60}")
    print(f"Environment: {name}")
    print(f"{'='*60}")
    print(f"Episode length: {env.get_wrapper_attr('timestep_per_episode')} timesteps")
    print(f"Runperiod: {env.get_wrapper_attr('runperiod')}")
    print(f"Observation space: {env.observation_space.shape[0]} variables")
    print(f"Action space: {env.action_space.shape[0]} variables")

def train_ppo_agent(env, total_timesteps=10000, name="Custom Agent"):
    """Train a PPO agent on the given environment."""
    print(f"\nTraining {name}...")
    
    # Create evaluation environment
    eval_env = create_custom_env(env.unwrapped.spec.id, env.unwrapped.config)
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{name}/",
        log_path=f"./logs/{name}/",
        eval_freq=2000,
        n_eval_episodes=2,
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
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    eval_env.close()
    return model

def example_1_summer_training():
    """Example 1: Summer-only training with energy focus."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Summer Training with Energy Focus")
    print("="*80)
    
    env_params = {
        # Summer months only (June 1 to August 31)
        'runperiod': [1, 6, 1991, 31, 8, 1991],
        'timesteps_per_hour': 4,
        
        # Energy-focused reward
        'reward': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': [20.0, 23.5],
            'range_comfort_summer': [23.0, 26.0],
            'summer_start': [6, 1],
            'summer_final': [8, 31],
            'energy_weight': 0.8,        # High energy focus
            'lambda_energy': 0.0001,
            'lambda_temperature': 0.5     # Lower comfort penalty
        }
    }
    
    env = create_custom_env('Eplus-5zone-hot-continuous-v1', env_params)
    print_env_info(env, "Summer Energy-Focused Training")
    
    # Train agent
    model = train_ppo_agent(env, total_timesteps=15000, name="summer_energy_focused")
    
    env.close()
    return model

def example_2_winter_training():
    """Example 2: Winter-only training with comfort focus."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Winter Training with Comfort Focus")
    print("="*80)
    
    env_params = {
        # Winter months only (December 1 to February 28)
        'runperiod': [1, 12, 1991, 28, 2, 1992],
        'timesteps_per_hour': 4,
        
        # Comfort-focused reward
        'reward': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': [20.0, 23.5],
            'range_comfort_summer': [23.0, 26.0],
            'summer_start': [6, 1],
            'summer_final': [8, 31],
            'energy_weight': 0.2,        # Low energy focus
            'lambda_energy': 0.0001,
            'lambda_temperature': 2.0     # Higher comfort penalty
        }
    }
    
    env = create_custom_env('Eplus-5zone-hot-continuous-v1', env_params)
    print_env_info(env, "Winter Comfort-Focused Training")
    
    # Train agent
    model = train_ppo_agent(env, total_timesteps=15000, name="winter_comfort_focused")
    
    env.close()
    return model

def example_3_full_year_balanced():
    """Example 3: Full year training with balanced approach."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Full Year Balanced Training")
    print("="*80)
    
    env_params = {
        # Full year simulation
        'runperiod': [1, 1, 1991, 31, 12, 1991],
        'timesteps_per_hour': 4,
        
        # Balanced reward
        'reward': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': [20.0, 23.5],
            'range_comfort_summer': [23.0, 26.0],
            'summer_start': [6, 1],
            'summer_final': [9, 30],
            'energy_weight': 0.5,        # Balanced
            'lambda_energy': 0.0001,
            'lambda_temperature': 1.0     # Standard comfort penalty
        }
    }
    
    env = create_custom_env('Eplus-5zone-hot-continuous-v1', env_params)
    print_env_info(env, "Full Year Balanced Training")
    
    # Train agent
    model = train_ppo_agent(env, total_timesteps=20000, name="full_year_balanced")
    
    env.close()
    return model

def example_4_custom_comfort_ranges():
    """Example 4: Custom comfort ranges for different seasons."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Comfort Ranges")
    print("="*80)
    
    env_params = {
        # Spring and fall months
        'runperiod': [1, 3, 1991, 31, 5, 1991],
        'timesteps_per_hour': 4,
        
        # Custom comfort ranges
        'reward': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': [18.0, 22.0],  # Stricter winter comfort
            'range_comfort_summer': [24.0, 28.0],  # Wider summer comfort
            'summer_start': [6, 1],
            'summer_final': [9, 30],
            'energy_weight': 0.6,        # Moderate energy focus
            'lambda_energy': 0.0001,
            'lambda_temperature': 1.5     # Higher comfort penalty
        }
    }
    
    env = create_custom_env('Eplus-5zone-hot-continuous-v1', env_params)
    print_env_info(env, "Custom Comfort Ranges Training")
    
    # Train agent
    model = train_ppo_agent(env, total_timesteps=12000, name="custom_comfort_ranges")
    
    env.close()
    return model

def example_5_multiple_energy_variables():
    """Example 5: Multiple energy variables in reward."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Multiple Energy Variables")
    print("="*80)
    
    env_params = {
        # Summer months
        'runperiod': [1, 6, 1991, 31, 8, 1991],
        'timesteps_per_hour': 4,
        
        # Multiple energy variables
        'reward': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': [
                'HVAC_electricity_demand_rate',
                'HVAC_gas_demand_rate',
                'total_electricity_demand_rate'
            ],
            'range_comfort_winter': [20.0, 23.5],
            'range_comfort_summer': [23.0, 26.0],
            'summer_start': [6, 1],
            'summer_final': [8, 31],
            'energy_weight': 0.7,        # High energy focus
            'lambda_energy': 0.0001,
            'lambda_temperature': 0.8     # Moderate comfort penalty
        }
    }
    
    env = create_custom_env('Eplus-5zone-hot-continuous-v1', env_params)
    print_env_info(env, "Multiple Energy Variables Training")
    
    # Train agent
    model = train_ppo_agent(env, total_timesteps=15000, name="multiple_energy_vars")
    
    env.close()
    return model

def compare_configurations():
    """Compare different configurations side by side."""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    configurations = {
        "Summer Energy-Focused": {
            'runperiod': [1, 6, 1991, 31, 8, 1991],
            'energy_weight': 0.8,
            'lambda_temperature': 0.5
        },
        "Winter Comfort-Focused": {
            'runperiod': [1, 12, 1991, 28, 2, 1992],
            'energy_weight': 0.2,
            'lambda_temperature': 2.0
        },
        "Full Year Balanced": {
            'runperiod': [1, 1, 1991, 31, 12, 1991],
            'energy_weight': 0.5,
            'lambda_temperature': 1.0
        },
        "Custom Comfort Ranges": {
            'runperiod': [1, 3, 1991, 31, 5, 1991],
            'energy_weight': 0.6,
            'lambda_temperature': 1.5
        }
    }
    
    print(f"{'Configuration':<25} {'Runperiod':<20} {'Energy Weight':<15} {'Comfort Penalty':<15}")
    print("-" * 80)
    
    for name, config in configurations.items():
        runperiod = config['runperiod']
        runperiod_str = f"{runperiod[1]}/{runperiod[0]}-{runperiod[4]}/{runperiod[3]}"
        print(f"{name:<25} {runperiod_str:<20} {config['energy_weight']:<15} {config['lambda_temperature']:<15}")

def main():
    """Run all customization examples."""
    print("Sinergym Customization Examples")
    print("="*80)
    print("This script demonstrates various ways to customize Sinergym environments")
    print("for different training scenarios.")
    
    # Compare configurations first
    compare_configurations()
    
    # Run examples (commented out to avoid long training times)
    # Uncomment the examples you want to run
    
    # Example 1: Summer training with energy focus
    # model1 = example_1_summer_training()
    
    # Example 2: Winter training with comfort focus  
    # model2 = example_2_winter_training()
    
    # Example 3: Full year balanced training
    # model3 = example_3_full_year_balanced()
    
    # Example 4: Custom comfort ranges
    # model4 = example_4_custom_comfort_ranges()
    
    # Example 5: Multiple energy variables
    # model5 = example_5_multiple_energy_variables()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print("To run the training examples, uncomment the desired examples in main().")
    print("Each example demonstrates different customization approaches:")
    print("1. Summer training with energy focus")
    print("2. Winter training with comfort focus")
    print("3. Full year balanced training")
    print("4. Custom comfort ranges")
    print("5. Multiple energy variables")

if __name__ == "__main__":
    main()