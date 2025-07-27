#!/usr/bin/env python3
"""
Test Different Reward Parameters for 5Zone Environment
======================================================

This script allows you to easily test different reward parameters
and see how they affect the reward calculation.
"""

import numpy as np
import sinergym
from sinergym.utils.rewards import LinearReward

def test_reward_parameters():
    """
    Test different reward parameter combinations.
    """
    print("🧪 TESTING REWARD PARAMETERS")
    print("=" * 50)
    
    # Test observations (different scenarios)
    test_observations = [
        {
            'name': 'Comfortable, Low Energy',
            'air_temperature': 22.0,  # Comfortable temperature
            'HVAC_electricity_demand_rate': 3000.0,  # Low energy consumption
            'month': 4,  # Spring
            'hour': 12
        },
        {
            'name': 'Too Hot, High Energy',
            'air_temperature': 28.0,  # Too hot
            'HVAC_electricity_demand_rate': 8000.0,  # High energy consumption
            'month': 7,  # Summer
            'hour': 14
        },
        {
            'name': 'Too Cold, High Energy',
            'air_temperature': 18.0,  # Too cold
            'HVAC_electricity_demand_rate': 6000.0,  # High energy consumption
            'month': 12,  # Winter
            'hour': 8
        },
        {
            'name': 'Comfortable, Moderate Energy',
            'air_temperature': 24.0,  # Comfortable temperature
            'HVAC_electricity_demand_rate': 5000.0,  # Moderate energy consumption
            'month': 5,  # Spring
            'hour': 16
        }
    ]
    
    # Different parameter combinations to test
    parameter_combinations = [
        {
            'name': 'Default (Balanced)',
            'energy_weight': 0.5,
            'lambda_energy': 1.0e-4,
            'lambda_temperature': 1.0
        },
        {
            'name': 'Energy Focused',
            'energy_weight': 0.8,
            'lambda_energy': 2.0e-4,
            'lambda_temperature': 0.5
        },
        {
            'name': 'Comfort Focused',
            'energy_weight': 0.2,
            'lambda_energy': 0.5e-4,
            'lambda_temperature': 2.0
        },
        {
            'name': 'High Energy Penalty',
            'energy_weight': 0.6,
            'lambda_energy': 3.0e-4,
            'lambda_temperature': 1.0
        },
        {
            'name': 'High Comfort Penalty',
            'energy_weight': 0.4,
            'lambda_energy': 1.0e-4,
            'lambda_temperature': 3.0
        }
    ]
    
    # Test each parameter combination
    for params in parameter_combinations:
        print(f"\n📊 {params['name']}")
        print("-" * 40)
        print(f"Energy weight: {params['energy_weight']}")
        print(f"Lambda energy: {params['lambda_energy']}")
        print(f"Lambda temperature: {params['lambda_temperature']}")
        print()
        
        # Create reward function with these parameters
        reward_fn = LinearReward(
            temperature_variables=['air_temperature'],
            energy_variables=['HVAC_electricity_demand_rate'],
            range_comfort_winter=(20.0, 23.5),
            range_comfort_summer=(23.0, 26.0),
            summer_start=(6, 1),
            summer_final=(9, 30),
            energy_weight=params['energy_weight'],
            lambda_energy=params['lambda_energy'],
            lambda_temperature=params['lambda_temperature']
        )
        
        # Test with each observation
        total_reward = 0
        for obs_data in test_observations:
            # Create observation dict
            obs = {
                'air_temperature': obs_data['air_temperature'],
                'HVAC_electricity_demand_rate': obs_data['HVAC_electricity_demand_rate'],
                'month': obs_data['month'],
                'hour': obs_data['hour']
            }
            
            # Calculate reward
            reward, reward_terms = reward_fn(obs)
            total_reward += reward
            
            print(f"  {obs_data['name']}:")
            print(f"    Temp: {obs_data['air_temperature']}°C, Energy: {obs_data['HVAC_electricity_demand_rate']}W")
            print(f"    Reward: {reward:.4f}")
            if reward_terms:
                print(f"    Terms: {reward_terms}")
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Average reward: {total_reward/len(test_observations):.4f}")

def create_custom_environment_with_reward():
    """
    Create an environment with custom reward parameters.
    """
    print("\n🔧 CREATING CUSTOM ENVIRONMENT")
    print("=" * 40)
    
    # Custom reward function
    custom_reward = LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(20.0, 23.5),
        range_comfort_summer=(23.0, 26.0),
        summer_start=(6, 1),
        summer_final=(9, 30),
        energy_weight=0.7,  # 70% energy focus
        lambda_energy=1.5e-4,  # Higher energy penalty
        lambda_temperature=0.6  # Lower comfort penalty
    )
    
    print("✅ Custom reward function created:")
    print(f"   Energy weight: 0.7")
    print(f"   Lambda energy: 1.5e-4")
    print(f"   Lambda temperature: 0.6")
    
    # Create environment with custom reward
    env = gym.make(
        'Eplus-5zone-hot-continuous-v1',
        env_name='CustomReward_Test',
        reward=custom_reward,
        config_params={
            'runperiod': (1, 1, 1991, 1, 3, 1991),  # 3 days
            'timesteps_per_hour': 4  # 15-minute timesteps
        }
    )
    
    print("✅ Custom environment created")
    print(f"   Run period: 3 days")
    print(f"   Timestep: 15 minutes")
    print(f"   Total timesteps: {env.timestep_per_episode}")
    
    # Test the environment
    print("\n🧪 Testing custom environment...")
    obs, info = env.reset()
    
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        print(f"  Step {step + 1}:")
        print(f"    Action: ({action[0]:.1f}, {action[1]:.1f})")
        print(f"    Air temp: {info.get('air_temperature', 'N/A'):.1f}°C")
        print(f"    HVAC power: {info.get('HVAC_electricity_demand_rate', 'N/A'):.2f}W")
        print(f"    Reward: {reward:.4f}")
    
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward: {total_reward/5:.4f}")
    
    env.close()
    print("✅ Environment closed")

def main():
    """
    Main function to run the reward parameter tests.
    """
    print("🎯 REWARD PARAMETER TESTING FOR 5ZONE ENVIRONMENT")
    print("=" * 60)
    
    try:
        import gymnasium as gym
        print("✅ Gymnasium imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test different reward parameters
    test_reward_parameters()
    
    # Create custom environment
    create_custom_environment_with_reward()
    
    print("\n🎉 REWARD PARAMETER TESTING COMPLETED!")
    print("=" * 60)
    print("📚 What you've learned:")
    print("  ✅ How different energy_weight affects rewards")
    print("  ✅ How lambda_energy changes energy penalties")
    print("  ✅ How lambda_temperature changes comfort penalties")
    print("  ✅ How to create custom environments with custom rewards")
    
    print("\n💡 Tips for choosing parameters:")
    print("  - energy_weight: 0.0-1.0 (higher = more energy focus)")
    print("  - lambda_energy: 1e-5 to 1e-3 (higher = stronger energy penalty)")
    print("  - lambda_temperature: 0.1-5.0 (higher = stronger comfort penalty)")

if __name__ == "__main__":
    main()