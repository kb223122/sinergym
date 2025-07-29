#!/usr/bin/env python3
"""
Final Corrected Solution for Sinergym PPO Training
==================================================

This fixes the parameter passing issue by using the correct parameter name 'extra_config'
instead of 'config_params'.
"""

import gymnasium as gym
import sinergym
from stable_baselines3 import PPO

def verify_parameters(env, expected_config, expected_reward):
    """Verify that parameters were applied correctly."""
    print(f"\n{'='*80}")
    print(f"PARAMETER VERIFICATION")
    print(f"{'='*80}")
    
    # Check runperiod
    actual_runperiod = env.get_wrapper_attr('runperiod')
    expected_runperiod = expected_config['runperiod']
    
    print(f"Expected runperiod: {expected_runperiod}")
    print(f"Actual runperiod: {actual_runperiod}")
    
    # Check reward function parameters
    reward_func = env.unwrapped.reward
    print(f"Reward function type: {type(reward_func).__name__}")
    
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

def train_with_verification():
    """Complete training with parameter verification."""
    
    # ✅ CORRECT: Use 'extra_config' instead of 'config_params'
    extra_config = {
        'runperiod': (1, 6, 1991, 31, 8, 1991),  # Summer months
        'timesteps_per_hour': 4
    }
    
    reward_kwargs = {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': (20.0, 23.5),
        'range_comfort_summer': (23.0, 26.0),
        'summer_start': (6, 1),
        'summer_final': (8, 31),
        'energy_weight': 0.7,
        'lambda_energy': 0.0001,
        'lambda_temperature': 0.8
    }
    
    print("Creating environment with custom parameters...")
    
    # ✅ CORRECT: Use 'extra_config' parameter
    env = gym.make('Eplus-5zone-hot-continuous-v1', 
                   extra_config=extra_config, 
                   reward_kwargs=reward_kwargs)
    
    # Print initial environment information
    print(f"Episode length: {env.get_wrapper_attr('timestep_per_episode')} timesteps")
    print(f"Runperiod: {env.get_wrapper_attr('runperiod')}")
    
    # Verify parameters were applied correctly
    verify_parameters(env, extra_config, reward_kwargs)
    
    # Episode-wise training
    timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
    num_episodes = 3
    total_timesteps = num_episodes * timesteps_per_episode
    
    print(f"\nTraining for {num_episodes} episodes ({total_timesteps} timesteps)")
    
    # Train PPO agent
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    env.close()
    return model

def step_by_step_verification(env, num_steps=5):
    """Test environment step by step to verify parameters."""
    print(f"\n{'='*80}")
    print(f"STEP-BY-STEP PARAMETER VERIFICATION")
    print(f"{'='*80}")
    
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(num_steps):
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Energy weight: {env.unwrapped.reward.W_energy}")
        print(f"  Temperature penalty: {env.unwrapped.reward.lambda_temp}")
        print(f"  Runperiod: {env.get_wrapper_attr('runperiod')}")
        
        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps")
            break
    
    print(f"\nTotal episode reward: {episode_reward:.4f}")

def main():
    """Main function with corrected parameter passing."""
    
    # ✅ CORRECT: Use 'extra_config' instead of 'config_params'
    extra_config = {
        'runperiod': (1, 6, 1991, 31, 8, 1991),  # Summer months
        'timesteps_per_hour': 4
    }
    
    reward_kwargs = {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': (20.0, 23.5),
        'range_comfort_summer': (23.0, 26.0),
        'summer_start': (6, 1),
        'summer_final': (8, 31),
        'energy_weight': 0.7,
        'lambda_energy': 0.0001,
        'lambda_temperature': 0.8
    }
    
    print("Creating environment with custom parameters...")
    
    # ✅ CORRECT: Use 'extra_config' parameter
    env = gym.make('Eplus-5zone-hot-continuous-v1', 
                   extra_config=extra_config, 
                   reward_kwargs=reward_kwargs)
    
    # Print initial environment information
    print(f"Episode length: {env.get_wrapper_attr('timestep_per_episode')} timesteps")
    print(f"Runperiod: {env.get_wrapper_attr('runperiod')}")
    
    # Verify parameters were applied correctly
    verify_parameters(env, extra_config, reward_kwargs)
    
    # Test environment step by step
    step_by_step_verification(env, num_steps=5)
    
    # Episode-wise training
    timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
    num_episodes = 3
    total_timesteps = num_episodes * timesteps_per_episode
    
    print(f"\nTraining for {num_episodes} episodes ({total_timesteps} timesteps)")
    
    # Train PPO agent
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    env.close()

if __name__ == "__main__":
    main()