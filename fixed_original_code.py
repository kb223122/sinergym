#!/usr/bin/env python3
"""
Fixed Original Code - Sinergym PPO Training
===========================================

This fixes the original code by passing parameters correctly through config_params
and reward_kwargs instead of directly as environment parameters.
"""

import gymnasium as gym
import sinergym
from stable_baselines3 import PPO

# FIXED: Custom environment configuration
# The runperiod and timesteps_per_hour should be passed through config_params
config_params = {
    # Time period: Summer months only
    'runperiod': (1, 6, 1991, 31, 8, 1991),  # June 1 to August 31, 1991
    
    # Simulation resolution
    'timesteps_per_hour': 4
}

# FIXED: Custom reward function parameters should be passed through reward_kwargs
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

# FIXED: Create environment with correct parameter passing
env = gym.make('Eplus-5zone-hot-continuous-v1', 
               config_params=config_params, 
               reward_kwargs=reward_kwargs)

# Print configuration to verify parameters were applied
print(f"Episode length: {env.get_wrapper_attr('timestep_per_episode')} timesteps")
print(f"Runperiod: {env.get_wrapper_attr('runperiod')}")

# Verify reward function parameters
reward_func = env.unwrapped.reward
print(f"Reward function type: {type(reward_func).__name__}")
if hasattr(reward_func, 'W_energy'):
    print(f"Energy weight: {reward_func.W_energy}")
if hasattr(reward_func, 'lambda_temp'):
    print(f"Temperature penalty: {reward_func.lambda_temp}")

# Train PPO agent episode-wise
# Calculate timesteps for 3 episodes
timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
total_timesteps = 3 * timesteps_per_episode  # 3 episodes

print(f"Training for 3 episodes ({total_timesteps} timesteps)")

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=total_timesteps)

print("Training completed!")
env.close()