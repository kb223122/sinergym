#!/usr/bin/env python3
"""
Comprehensive 5Zone Hot Environment Customization Guide
=======================================================

This script demonstrates ALL possible customizations for the 
'Eplus-5zone-hot-continuous-v1' environment in Sinergym.

What you'll learn:
1. How to change reward parameters (lambda_temp, energy_weight, lambda_energy)
2. How to modify environment run periods and timesteps
3. How to use different reward functions
4. How to customize observation and action spaces
5. How to apply weather variability
6. How to monitor and verify all changes

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# =============================================================================
# STEP 1: IMPORTS AND SETUP
# =============================================================================

print("🚀 Starting Comprehensive 5Zone Customization Guide")
print("=" * 60)

try:
    import gymnasium as gym
    import sinergym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from sinergym.utils.wrappers import (
        NormalizeObservation, NormalizeAction, 
        LoggerWrapper, CSVLogger, WandBLogger
    )
    from sinergym.utils.rewards import (
        LinearReward, ExpReward, EnergyCostLinearReward,
        HourlyLinearReward, NormalizedLinearReward, MultiZoneReward
    )
    from sinergym.utils.callbacks import LoggerEvalCallback
    print("✅ All libraries imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install sinergym[drl]")
    sys.exit(1)

# =============================================================================
# STEP 2: ENVIRONMENT ANALYSIS FUNCTIONS
# =============================================================================

def analyze_environment(env, env_name: str):
    """
    Analyze and print detailed information about the environment.
    """
    print(f"\n🔍 ENVIRONMENT ANALYSIS: {env_name}")
    print("-" * 50)
    
    # Basic environment info
    print(f"Environment Name: {env.name}")
    print(f"Environment ID: {env.unwrapped.spec.id}")
    
    # Observation space
    obs_space = env.observation_space
    print(f"\n📊 OBSERVATION SPACE:")
    print(f"  Type: {type(obs_space)}")
    print(f"  Shape: {obs_space.shape}")
    print(f"  Low: {obs_space.low}")
    print(f"  High: {obs_space.high}")
    print(f"  Number of variables: {obs_space.shape[0]}")
    
    # Action space
    action_space = env.action_space
    print(f"\n🎯 ACTION SPACE:")
    print(f"  Type: {type(action_space)}")
    print(f"  Shape: {action_space.shape}")
    print(f"  Low: {action_space.low}")
    print(f"  High: {action_space.high}")
    print(f"  Action 1 (Heating Setpoint): {action_space.low[0]}°C to {action_space.high[0]}°C")
    print(f"  Action 2 (Cooling Setpoint): {action_space.low[1]}°C to {action_space.high[1]}°C")
    
    # Episode information
    print(f"\n⏰ EPISODE INFORMATION:")
    print(f"  Episode length: {env.episode_length} seconds")
    print(f"  Timestep size: {env.step_size} seconds")
    print(f"  Timesteps per episode: {env.timestep_per_episode}")
    print(f"  Episode duration: {env.episode_length / 3600:.1f} hours")
    print(f"  Episode duration: {env.episode_length / (3600 * 24):.1f} days")
    
    # Run period
    runperiod = env.runperiod
    print(f"\n📅 RUN PERIOD:")
    print(f"  Start: {runperiod['start_day']}/{runperiod['start_month']}/{runperiod['start_year']}")
    print(f"  End: {runperiod['end_day']}/{runperiod['end_month']}/{runperiod['end_year']}")
    
    # Building information
    print(f"\n🏢 BUILDING INFORMATION:")
    print(f"  Building file: {env.building_file}")
    print(f"  Weather file: {env.weather_path}")
    print(f"  Zone names: {env.zone_names}")
    
    # Variables and meters
    print(f"\n📈 MONITORED VARIABLES:")
    for var_name, (var_key, var_obj) in env.variables.items():
        print(f"  {var_name}: {var_key} -> {var_obj}")
    
    print(f"\n⚡ MONITORED METERS:")
    for meter_name, meter_obj in env.meters.items():
        print(f"  {meter_name}: {meter_obj}")
    
    # Actuators
    print(f"\n🎛️ ACTUATORS:")
    for actuator_name, (actuator_type, value_type, actuator_obj) in env.actuators.items():
        print(f"  {actuator_name}: {actuator_type} -> {value_type} -> {actuator_obj}")

def test_environment_step(env, num_steps: int = 5):
    """
    Test the environment by taking a few steps and showing the data.
    """
    print(f"\n🧪 TESTING ENVIRONMENT ({num_steps} steps)")
    print("-" * 40)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info keys: {list(info.keys())}")
    
    total_reward = 0
    step_data = []
    
    for step in range(num_steps):
        # Take random action
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store data
        step_data.append({
            'step': step + 1,
            'action_heating': action[0],
            'action_cooling': action[1],
            'reward': reward,
            'air_temperature': info.get('air_temperature', 'N/A'),
            'outdoor_temperature': info.get('outdoor_temperature', 'N/A'),
            'HVAC_electricity': info.get('HVAC_electricity_demand_rate', 'N/A'),
            'month': info.get('month', 'N/A'),
            'hour': info.get('hour', 'N/A')
        })
        
        total_reward += reward
        
        print(f"Step {step + 1}:")
        print(f"  Action: Heating={action[0]:.1f}°C, Cooling={action[1]:.1f}°C")
        print(f"  Reward: {reward:.4f}")
        print(f"  Air Temp: {info.get('air_temperature', 'N/A'):.1f}°C")
        print(f"  Outdoor Temp: {info.get('outdoor_temperature', 'N/A'):.1f}°C")
        print(f"  HVAC Power: {info.get('HVAC_electricity_demand_rate', 'N/A'):.2f} W")
        print(f"  Time: Month {info.get('month', 'N/A')}, Hour {info.get('hour', 'N/A')}")
        print()
    
    print(f"Total reward over {num_steps} steps: {total_reward:.4f}")
    print(f"Average reward per step: {total_reward/num_steps:.4f}")
    
    return step_data

# =============================================================================
# STEP 3: REWARD FUNCTION CUSTOMIZATION
# =============================================================================

def create_custom_reward_functions():
    """
    Create and demonstrate different reward functions available in Sinergym.
    """
    print("\n🎯 REWARD FUNCTION CUSTOMIZATION")
    print("=" * 50)
    
    # Base parameters
    temperature_variables = ['air_temperature']
    energy_variables = ['HVAC_electricity_demand_rate']
    range_comfort_winter = (20.0, 23.5)
    range_comfort_summer = (23.0, 26.0)
    summer_start = (6, 1)
    summer_final = (9, 30)
    
    reward_functions = {}
    
    # 1. Linear Reward (Default)
    print("\n1️⃣ LINEAR REWARD (Default)")
    print("-" * 30)
    linear_reward = LinearReward(
        temperature_variables=temperature_variables,
        energy_variables=energy_variables,
        range_comfort_winter=range_comfort_winter,
        range_comfort_summer=range_comfort_summer,
        summer_start=summer_start,
        summer_final=summer_final,
        energy_weight=0.5,
        lambda_energy=1.0e-4,
        lambda_temperature=1.0
    )
    reward_functions['LinearReward'] = linear_reward
    print("✅ Linear reward function created")
    print("   Formula: R = -W * λ_E * Energy - (1-W) * λ_T * Temperature_Violation")
    print(f"   Energy weight (W): 0.5")
    print(f"   Lambda energy (λ_E): 1.0e-4")
    print(f"   Lambda temperature (λ_T): 1.0")
    
    # 2. Custom Linear Reward with different weights
    print("\n2️⃣ CUSTOM LINEAR REWARD (Energy Focused)")
    print("-" * 40)
    energy_focused_reward = LinearReward(
        temperature_variables=temperature_variables,
        energy_variables=energy_variables,
        range_comfort_winter=range_comfort_winter,
        range_comfort_summer=range_comfort_summer,
        summer_start=summer_start,
        summer_final=summer_final,
        energy_weight=0.8,  # Higher energy weight
        lambda_energy=2.0e-4,  # Higher energy penalty
        lambda_temperature=0.5  # Lower temperature penalty
    )
    reward_functions['EnergyFocusedReward'] = energy_focused_reward
    print("✅ Energy-focused reward function created")
    print(f"   Energy weight (W): 0.8 (higher)")
    print(f"   Lambda energy (λ_E): 2.0e-4 (higher penalty)")
    print(f"   Lambda temperature (λ_T): 0.5 (lower penalty)")
    
    # 3. Comfort Focused Reward
    print("\n3️⃣ COMFORT FOCUSED REWARD")
    print("-" * 30)
    comfort_focused_reward = LinearReward(
        temperature_variables=temperature_variables,
        energy_variables=energy_variables,
        range_comfort_winter=range_comfort_winter,
        range_comfort_summer=range_comfort_summer,
        summer_start=summer_start,
        summer_final=summer_final,
        energy_weight=0.2,  # Lower energy weight
        lambda_energy=0.5e-4,  # Lower energy penalty
        lambda_temperature=2.0  # Higher temperature penalty
    )
    reward_functions['ComfortFocusedReward'] = comfort_focused_reward
    print("✅ Comfort-focused reward function created")
    print(f"   Energy weight (W): 0.2 (lower)")
    print(f"   Lambda energy (λ_E): 0.5e-4 (lower penalty)")
    print(f"   Lambda temperature (λ_T): 2.0 (higher penalty)")
    
    # 4. Exponential Reward
    print("\n4️⃣ EXPONENTIAL REWARD")
    print("-" * 25)
    exp_reward = ExpReward(
        temperature_variables=temperature_variables,
        energy_variables=energy_variables,
        range_comfort_winter=range_comfort_winter,
        range_comfort_summer=range_comfort_summer,
        summer_start=summer_start,
        summer_final=summer_final,
        energy_weight=0.5,
        lambda_energy=1.0e-4,
        lambda_temperature=1.0
    )
    reward_functions['ExpReward'] = exp_reward
    print("✅ Exponential reward function created")
    print("   Formula: Uses exponential penalties for violations")
    
    # 5. Hourly Linear Reward
    print("\n5️⃣ HOURLY LINEAR REWARD")
    print("-" * 30)
    hourly_reward = HourlyLinearReward(
        temperature_variables=temperature_variables,
        energy_variables=energy_variables,
        range_comfort_winter=range_comfort_winter,
        range_comfort_summer=range_comfort_summer,
        summer_start=summer_start,
        summer_final=summer_final,
        default_energy_weight=0.5,
        lambda_energy=1.0e-4,
        lambda_temperature=1.0,
        range_comfort_hours=(9, 19)  # Comfort only during 9 AM to 7 PM
    )
    reward_functions['HourlyLinearReward'] = hourly_reward
    print("✅ Hourly linear reward function created")
    print("   Comfort hours: 9 AM to 7 PM only")
    print("   Outside hours: Only energy optimization")
    
    # 6. Normalized Linear Reward
    print("\n6️⃣ NORMALIZED LINEAR REWARD")
    print("-" * 35)
    normalized_reward = NormalizedLinearReward(
        temperature_variables=temperature_variables,
        energy_variables=energy_variables,
        range_comfort_winter=range_comfort_winter,
        range_comfort_summer=range_comfort_summer,
        summer_start=summer_start,
        summer_final=summer_final,
        energy_weight=0.5,
        max_energy_penalty=8,
        max_comfort_penalty=12
    )
    reward_functions['NormalizedLinearReward'] = normalized_reward
    print("✅ Normalized linear reward function created")
    print("   Max energy penalty: 8")
    print("   Max comfort penalty: 12")
    print("   Rewards are normalized to these maximum values")
    
    return reward_functions

def test_reward_functions(reward_functions: Dict, test_observations: List[Dict]):
    """
    Test different reward functions with sample observations.
    """
    print("\n🧪 TESTING REWARD FUNCTIONS")
    print("=" * 40)
    
    for reward_name, reward_fn in reward_functions.items():
        print(f"\n📊 Testing {reward_name}:")
        print("-" * 30)
        
        total_reward = 0
        for i, obs in enumerate(test_observations):
            reward, reward_terms = reward_fn(obs)
            total_reward += reward
            
            print(f"  Observation {i+1}:")
            print(f"    Air temp: {obs.get('air_temperature', 'N/A'):.1f}°C")
            print(f"    HVAC power: {obs.get('HVAC_electricity_demand_rate', 'N/A'):.2f} W")
            print(f"    Reward: {reward:.4f}")
            if reward_terms:
                print(f"    Reward terms: {reward_terms}")
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Average reward: {total_reward/len(test_observations):.4f}")

# =============================================================================
# STEP 4: ENVIRONMENT CONFIGURATION CUSTOMIZATION
# =============================================================================

def create_custom_environments():
    """
    Create environments with different configurations.
    """
    print("\n⚙️ ENVIRONMENT CONFIGURATION CUSTOMIZATION")
    print("=" * 55)
    
    environments = {}
    
    # 1. Default Environment
    print("\n1️⃣ DEFAULT ENVIRONMENT")
    print("-" * 25)
    default_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name='Default_5Zone_Hot')
    environments['default'] = default_env
    print("✅ Default environment created")
    
    # 2. Environment with Custom Run Period (1 month)
    print("\n2️⃣ CUSTOM RUN PERIOD (1 MONTH)")
    print("-" * 35)
    one_month_env = gym.make(
        'Eplus-5zone-hot-continuous-v1',
        env_name='OneMonth_5Zone_Hot',
        config_params={
            'runperiod': (1, 1, 1991, 1, 31, 1991),  # January 1-31, 1991
            'timesteps_per_hour': 1  # 1 timestep = 1 hour
        }
    )
    environments['one_month'] = one_month_env
    print("✅ One-month environment created")
    print("   Run period: January 1-31, 1991")
    print("   Timestep: 1 hour")
    print(f"   Total timesteps: {one_month_env.timestep_per_episode}")
    
    # 3. Environment with Custom Run Period (1 week)
    print("\n3️⃣ CUSTOM RUN PERIOD (1 WEEK)")
    print("-" * 35)
    one_week_env = gym.make(
        'Eplus-5zone-hot-continuous-v1',
        env_name='OneWeek_5Zone_Hot',
        config_params={
            'runperiod': (1, 1, 1991, 1, 7, 1991),  # January 1-7, 1991
            'timesteps_per_hour': 4  # 1 timestep = 15 minutes
        }
    )
    environments['one_week'] = one_week_env
    print("✅ One-week environment created")
    print("   Run period: January 1-7, 1991")
    print("   Timestep: 15 minutes")
    print(f"   Total timesteps: {one_week_env.timestep_per_episode}")
    
    # 4. Environment with Custom Run Period (1 day)
    print("\n4️⃣ CUSTOM RUN PERIOD (1 DAY)")
    print("-" * 35)
    one_day_env = gym.make(
        'Eplus-5zone-hot-continuous-v1',
        env_name='OneDay_5Zone_Hot',
        config_params={
            'runperiod': (1, 1, 1991, 1, 1, 1991),  # January 1, 1991
            'timesteps_per_hour': 12  # 1 timestep = 5 minutes
        }
    )
    environments['one_day'] = one_day_env
    print("✅ One-day environment created")
    print("   Run period: January 1, 1991")
    print("   Timestep: 5 minutes")
    print(f"   Total timesteps: {one_day_env.timestep_per_episode}")
    
    # 5. Environment with Weather Variability
    print("\n5️⃣ WEATHER VARIABILITY ENVIRONMENT")
    print("-" * 40)
    weather_var_env = gym.make(
        'Eplus-5zone-hot-continuous-v1',
        env_name='WeatherVar_5Zone_Hot',
        weather_variability={
            'Site Outdoor Air DryBulb Temperature': (2.0, 0.0, 24.0),  # sigma, mu, tau
            'Site Outdoor Air Relative Humidity': (5.0, 0.0, 24.0)
        }
    )
    environments['weather_var'] = weather_var_env
    print("✅ Weather variability environment created")
    print("   Temperature variability: σ=2.0°C, μ=0.0, τ=24h")
    print("   Humidity variability: σ=5.0%, μ=0.0, τ=24h")
    
    # 6. Environment with Custom Reward Function
    print("\n6️⃣ CUSTOM REWARD ENVIRONMENT")
    print("-" * 35)
    custom_reward = LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(20.0, 23.5),
        range_comfort_summer=(23.0, 26.0),
        summer_start=(6, 1),
        summer_final=(9, 30),
        energy_weight=0.7,  # Higher energy focus
        lambda_energy=1.5e-4,  # Higher energy penalty
        lambda_temperature=0.8  # Lower comfort penalty
    )
    
    custom_reward_env = gym.make(
        'Eplus-5zone-hot-continuous-v1',
        env_name='CustomReward_5Zone_Hot',
        reward=custom_reward
    )
    environments['custom_reward'] = custom_reward_env
    print("✅ Custom reward environment created")
    print("   Energy weight: 0.7 (higher)")
    print("   Lambda energy: 1.5e-4 (higher penalty)")
    print("   Lambda temperature: 0.8 (lower penalty)")
    
    return environments

# =============================================================================
# STEP 5: COMPREHENSIVE DEMONSTRATION
# =============================================================================

def run_comprehensive_demo():
    """
    Run a comprehensive demonstration of all customizations.
    """
    print("\n🎬 COMPREHENSIVE DEMONSTRATION")
    print("=" * 50)
    
    # Create reward functions
    reward_functions = create_custom_reward_functions()
    
    # Create test observations
    test_observations = [
        {
            'air_temperature': 22.0,  # Comfortable
            'HVAC_electricity_demand_rate': 5000.0,  # Moderate energy
            'month': 1,
            'hour': 12
        },
        {
            'air_temperature': 28.0,  # Too hot
            'HVAC_electricity_demand_rate': 8000.0,  # High energy
            'month': 7,
            'hour': 14
        },
        {
            'air_temperature': 18.0,  # Too cold
            'HVAC_electricity_demand_rate': 6000.0,  # High energy
            'month': 12,
            'hour': 8
        },
        {
            'air_temperature': 24.0,  # Comfortable
            'HVAC_electricity_demand_rate': 3000.0,  # Low energy
            'month': 4,
            'hour': 16
        }
    ]
    
    # Test reward functions
    test_reward_functions(reward_functions, test_observations)
    
    # Create custom environments
    environments = create_custom_environments()
    
    # Analyze each environment
    for env_name, env in environments.items():
        analyze_environment(env, f"Environment: {env_name}")
        
        # Test a few steps
        if env_name in ['one_day', 'one_week']:  # Test shorter environments
            test_environment_step(env, num_steps=3)
        else:
            test_environment_step(env, num_steps=2)
        
        env.close()
        print(f"\n✅ {env_name} environment tested and closed")
        print("-" * 60)

# =============================================================================
# STEP 6: PRACTICAL TRAINING EXAMPLE
# =============================================================================

def train_with_custom_configuration():
    """
    Train a PPO agent with custom configuration.
    """
    print("\n🚀 PRACTICAL TRAINING EXAMPLE")
    print("=" * 40)
    
    # Create custom environment with specific configuration
    print("\n📋 Creating custom training environment...")
    
    # Custom reward function
    custom_reward = LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(20.0, 23.5),
        range_comfort_summer=(23.0, 26.0),
        summer_start=(6, 1),
        summer_final=(9, 30),
        energy_weight=0.6,  # 60% energy focus, 40% comfort
        lambda_energy=1.2e-4,  # Energy penalty coefficient
        lambda_temperature=0.8  # Comfort penalty coefficient
    )
    
    # Create environment with 1-week run period
    train_env = gym.make(
        'Eplus-5zone-hot-continuous-v1',
        env_name='CustomTraining_5Zone_Hot',
        config_params={
            'runperiod': (1, 1, 1991, 1, 7, 1991),  # 1 week
            'timesteps_per_hour': 4  # 15-minute timesteps
        },
        reward=custom_reward
    )
    
    eval_env = gym.make(
        'Eplus-5zone-hot-continuous-v1',
        env_name='CustomEvaluation_5Zone_Hot',
        config_params={
            'runperiod': (1, 1, 1991, 1, 7, 1991),  # 1 week
            'timesteps_per_hour': 4  # 15-minute timesteps
        },
        reward=custom_reward
    )
    
    print("✅ Custom training environment created")
    print(f"   Run period: 1 week (January 1-7, 1991)")
    print(f"   Timestep: 15 minutes")
    print(f"   Total timesteps per episode: {train_env.timestep_per_episode}")
    print(f"   Energy weight: 0.6")
    print(f"   Lambda energy: 1.2e-4")
    print(f"   Lambda temperature: 0.8")
    
    # Add wrappers
    train_env = NormalizeObservation(train_env)
    train_env = NormalizeAction(train_env)
    train_env = LoggerWrapper(train_env)
    train_env = CSVLogger(train_env)
    
    eval_env = NormalizeObservation(eval_env)
    eval_env = NormalizeAction(eval_env)
    eval_env = LoggerWrapper(eval_env)
    eval_env = CSVLogger(eval_env)
    
    print("✅ Wrappers applied")
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=0.0003,
        n_steps=1024,  # Smaller for shorter episodes
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    print("✅ PPO model created")
    
    # Setup evaluation callback
    eval_callback = LoggerEvalCallback(
        eval_env=eval_env,
        train_env=train_env,
        n_eval_episodes=1,
        eval_freq_episodes=1,  # Evaluate every episode
        deterministic=True
    )
    
    print("✅ Evaluation callback created")
    
    # Calculate training timesteps (5 episodes)
    total_timesteps = 5 * train_env.timestep_per_episode
    print(f"\n🎯 Starting training for {total_timesteps} timesteps (5 episodes)")
    print("   This will take a few minutes...")
    
    # Train the model
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=100
    )
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    
    # Save the model
    model_path = 'custom_ppo_5zone_model'
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Evaluate the trained model
    print("\n📊 Evaluating trained model...")
    evaluate_trained_model(model, eval_env, num_episodes=2)
    
    # Clean up
    train_env.close()
    eval_env.close()
    print("✅ Environments closed")

def evaluate_trained_model(model, eval_env, num_episodes: int = 2):
    """
    Evaluate a trained model and show detailed results.
    """
    print(f"\n🔍 MODEL EVALUATION ({num_episodes} episodes)")
    print("-" * 45)
    
    episode_results = []
    
    for episode in range(num_episodes):
        print(f"\n📈 Episode {episode + 1}:")
        print("-" * 20)
        
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_energy = 0
        comfort_violations = 0
        total_steps = 0
        
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            episode_reward += reward
            episode_energy += info.get('HVAC_electricity_demand_rate', 0)
            
            # Check comfort violations
            temp = info.get('air_temperature', 0)
            if temp < 20.0 or temp > 26.0:
                comfort_violations += 1
            
            total_steps += 1
            
            # Print some steps for demonstration
            if total_steps <= 5 or total_steps % 50 == 0:
                print(f"  Step {total_steps}: Temp={temp:.1f}°C, "
                      f"Action=({action[0]:.1f}, {action[1]:.1f}), "
                      f"Reward={reward:.4f}")
        
        episode_results.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'total_energy': episode_energy,
            'comfort_violations': comfort_violations,
            'total_steps': total_steps,
            'avg_reward': episode_reward / total_steps,
            'avg_energy': episode_energy / total_steps,
            'violation_rate': comfort_violations / total_steps
        })
        
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Total energy: {episode_energy:.2f} W")
        print(f"  Comfort violations: {comfort_violations}/{total_steps} "
              f"({comfort_violations/total_steps*100:.1f}%)")
        print(f"  Average reward per step: {episode_reward/total_steps:.4f}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print("-" * 25)
    avg_reward = np.mean([r['total_reward'] for r in episode_results])
    avg_energy = np.mean([r['total_energy'] for r in episode_results])
    avg_violation_rate = np.mean([r['violation_rate'] for r in episode_results])
    
    print(f"  Average total reward: {avg_reward:.4f}")
    print(f"  Average total energy: {avg_energy:.2f} W")
    print(f"  Average comfort violation rate: {avg_violation_rate*100:.1f}%")
    
    return episode_results

# =============================================================================
# STEP 7: MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the comprehensive demonstration.
    """
    print("🎯 COMPREHENSIVE 5ZONE HOT ENVIRONMENT CUSTOMIZATION GUIDE")
    print("=" * 70)
    print(f"Sinergym Version: {sinergym.__version__}")
    print(f"Python Version: {sys.version}")
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Run comprehensive demo
        run_comprehensive_demo()
        
        # Ask user if they want to run training
        print("\n" + "=" * 70)
        response = input("\n🤔 Would you like to run the practical training example? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            train_with_custom_configuration()
        else:
            print("✅ Skipping training example")
        
        print("\n🎉 COMPREHENSIVE DEMONSTRATION COMPLETED!")
        print("=" * 70)
        print("📚 What you've learned:")
        print("  ✅ How to customize reward parameters")
        print("  ✅ How to change environment run periods")
        print("  ✅ How to use different reward functions")
        print("  ✅ How to apply weather variability")
        print("  ✅ How to train and evaluate PPO agents")
        print("  ✅ How to monitor and verify all changes")
        
        print("\n📁 Generated files:")
        print("  - custom_ppo_5zone_model.zip (if training was run)")
        print("  - Various CSV logs in environment directories")
        
        print("\n🚀 You're now ready to customize your own Sinergym environments!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting tips:")
        print("  - Check if EnergyPlus is properly installed")
        print("  - Verify all dependencies are installed: pip install sinergym[drl]")
        print("  - Ensure you have sufficient disk space for simulations")
        print("  - Check if the environment ID is correct")

if __name__ == "__main__":
    main()