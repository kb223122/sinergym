#!/usr/bin/env python3
"""
Demo: Correct Parameter Passing Approach
=======================================

This demo shows the correct way to pass parameters in Sinergym
without requiring EnergyPlus installation.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Create a simple custom environment to demonstrate parameter passing
class SimpleBuildingEnv(gym.Env):
    def __init__(self, config_params=None, reward_kwargs=None):
        super().__init__()
        
        # Store parameters for demonstration
        self.config_params = config_params or {}
        self.reward_kwargs = reward_kwargs or {}
        
        # Simulate environment properties
        self.timestep_per_episode = 8760  # Default episode length
        self.runperiod = {
            'start_month': 1,
            'start_day': 1,
            'end_month': 12,
            'end_day': 31,
            'start_year': 1991,
            'end_year': 1991
        }
        
        # Update runperiod if provided
        if config_params and 'runperiod' in config_params:
            runperiod = config_params['runperiod']
            self.runperiod = {
                'start_month': runperiod[1],
                'start_day': runperiod[0],
                'end_month': runperiod[4],
                'end_day': runperiod[3],
                'start_year': runperiod[2],
                'end_year': runperiod[5]
            }
        
        # Update timesteps if provided
        if config_params and 'timesteps_per_hour' in config_params:
            self.timestep_per_episode = 8760 * config_params['timesteps_per_hour'] // 4
        
        # Create reward function with provided parameters
        self.reward = SimpleReward(reward_kwargs or {})
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=-50, high=50, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=12, high=30, shape=(2,), dtype=np.float32)
        
        self.current_step = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.random.randn(10), {}
    
    def step(self, action):
        self.current_step += 1
        
        # Simulate observation
        obs = np.random.randn(10)
        
        # Calculate reward using the reward function
        reward = self.reward(obs, action)
        
        # Episode ends after timestep_per_episode steps
        done = self.current_step >= self.timestep_per_episode
        
        return obs, reward, done, False, {}
    
    def get_wrapper_attr(self, attr):
        """Simulate Sinergym wrapper attributes."""
        if attr == 'timestep_per_episode':
            return self.timestep_per_episode
        elif attr == 'runperiod':
            return self.runperiod
        return None

class SimpleReward:
    def __init__(self, kwargs):
        self.energy_weight = kwargs.get('energy_weight', 0.5)
        self.lambda_energy = kwargs.get('lambda_energy', 0.0001)
        self.lambda_temperature = kwargs.get('lambda_temperature', 1.0)
        self.temperature_variables = kwargs.get('temperature_variables', ['air_temperature'])
        self.energy_variables = kwargs.get('energy_variables', ['HVAC_electricity_demand_rate'])
        
    def __call__(self, obs, action):
        # Simulate reward calculation
        energy_penalty = -self.energy_weight * self.lambda_energy * abs(obs[-1])
        comfort_penalty = -(1 - self.energy_weight) * self.lambda_temperature * abs(obs[8] - 22.0)
        return energy_penalty + comfort_penalty

def demonstrate_parameter_passing():
    """Demonstrate the correct parameter passing approach."""
    
    print("="*80)
    print("DEMONSTRATING CORRECT PARAMETER PASSING")
    print("="*80)
    
    # Example 1: Default parameters
    print("\n1. Creating environment with default parameters...")
    env1 = SimpleBuildingEnv()
    print(f"   Default episode length: {env1.timestep_per_episode}")
    print(f"   Default runperiod: {env1.runperiod}")
    print(f"   Default energy weight: {env1.reward.energy_weight}")
    
    # Example 2: Custom runperiod
    print("\n2. Creating environment with custom runperiod...")
    config_params = {
        'runperiod': (1, 6, 1991, 31, 8, 1991),  # Summer only
        'timesteps_per_hour': 4
    }
    env2 = SimpleBuildingEnv(config_params=config_params)
    print(f"   Custom episode length: {env2.timestep_per_episode}")
    print(f"   Custom runperiod: {env2.runperiod}")
    
    # Example 3: Custom reward parameters
    print("\n3. Creating environment with custom reward parameters...")
    reward_kwargs = {
        'energy_weight': 0.8,
        'lambda_energy': 0.0001,
        'lambda_temperature': 0.5
    }
    env3 = SimpleBuildingEnv(reward_kwargs=reward_kwargs)
    print(f"   Custom energy weight: {env3.reward.energy_weight}")
    print(f"   Custom temperature penalty: {env3.reward.lambda_temperature}")
    
    # Example 4: Both custom config and reward
    print("\n4. Creating environment with both custom config and reward...")
    env4 = SimpleBuildingEnv(config_params=config_params, reward_kwargs=reward_kwargs)
    print(f"   Combined episode length: {env4.timestep_per_episode}")
    print(f"   Combined runperiod: {env4.runperiod}")
    print(f"   Combined energy weight: {env4.reward.energy_weight}")
    
    return env4

def train_episode_wise(env, num_episodes=3):
    """Train PPO agent episode-wise."""
    print(f"\n{'='*80}")
    print(f"EPISODE-WISE TRAINING")
    print(f"{'='*80}")
    
    # Calculate total timesteps for episodes
    timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
    total_timesteps = num_episodes * timesteps_per_episode
    
    print(f"Training for {num_episodes} episodes")
    print(f"Timesteps per episode: {timesteps_per_episode}")
    print(f"Total timesteps: {total_timesteps}")
    
    # Create and train model
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    return model

def verify_parameters_at_steps(env, num_steps=5):
    """Verify parameters at each step."""
    print(f"\n{'='*80}")
    print(f"PARAMETER VERIFICATION AT EACH STEP")
    print(f"{'='*80}")
    
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Energy weight: {env.reward.energy_weight}")
        print(f"  Temperature penalty: {env.reward.lambda_temperature}")
        print(f"  Runperiod: {env.runperiod}")
        
        if done:
            print(f"Episode ended after {step + 1} steps")
            break
    
    print(f"\nTotal episode reward: {episode_reward:.4f}")

def main():
    """Main demonstration function."""
    
    # Demonstrate correct parameter passing
    env = demonstrate_parameter_passing()
    
    # Verify parameters at each step
    verify_parameters_at_steps(env, num_steps=5)
    
    # Train episode-wise
    model = train_episode_wise(env, num_episodes=2)
    
    print(f"\n{'='*80}")
    print(f"DEMONSTRATION COMPLETED!")
    print(f"{'='*80}")
    print("Key points:")
    print("1. Use config_params for runperiod and timesteps_per_hour")
    print("2. Use reward_kwargs for reward function parameters")
    print("3. Episode-wise training: calculate timesteps = episodes × timesteps_per_episode")
    print("4. Always verify parameters are applied correctly")

if __name__ == "__main__":
    main()