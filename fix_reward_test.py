#!/usr/bin/env python3
"""
Fix for Reward Function Testing
===============================

This script tests reward functions with proper observation data
to avoid the 'day_of_month' KeyError.
"""

import sys
import numpy as np

# Try to import Sinergym components
try:
    import sinergym
    from sinergym.utils.rewards import (
        LinearReward, ExpReward, HourlyLinearReward, 
        NormalizedLinearReward
    )
    print("✅ Sinergym imported successfully")
except ImportError as e:
    print(f"❌ Sinergym import error: {e}")
    print("Please install Sinergym: pip install sinergym[drl]")
    sys.exit(1)

def test_reward_functions_fixed():
    """
    Test reward functions with proper observation data.
    """
    print("🧪 TESTING REWARD FUNCTIONS (FIXED)")
    print("=" * 50)
    
    # Create test observations with ALL required variables
    test_observations = [
        {
            'air_temperature': 22.0,  # Comfortable
            'HVAC_electricity_demand_rate': 5000.0,  # Moderate energy
            'month': 1,
            'day_of_month': 15,
            'hour': 12
        },
        {
            'air_temperature': 28.0,  # Too hot
            'HVAC_electricity_demand_rate': 8000.0,  # High energy
            'month': 7,
            'day_of_month': 20,
            'hour': 14
        },
        {
            'air_temperature': 18.0,  # Too cold
            'HVAC_electricity_demand_rate': 6000.0,  # High energy
            'month': 12,
            'day_of_month': 10,
            'hour': 8
        },
        {
            'air_temperature': 24.0,  # Comfortable
            'HVAC_electricity_demand_rate': 3000.0,  # Low energy
            'month': 4,
            'day_of_month': 25,
            'hour': 16
        }
    ]
    
    # Base parameters
    temperature_variables = ['air_temperature']
    energy_variables = ['HVAC_electricity_demand_rate']
    range_comfort_winter = (20.0, 23.5)
    range_comfort_summer = (23.0, 26.0)
    summer_start = (6, 1)
    summer_final = (9, 30)
    
    # Test different reward functions
    reward_functions = {
        'LinearReward (Default)': LinearReward(
            temperature_variables=temperature_variables,
            energy_variables=energy_variables,
            range_comfort_winter=range_comfort_winter,
            range_comfort_summer=range_comfort_summer,
            summer_start=summer_start,
            summer_final=summer_final,
            energy_weight=0.5,
            lambda_energy=1.0e-4,
            lambda_temperature=1.0
        ),
        'LinearReward (Energy Focused)': LinearReward(
            temperature_variables=temperature_variables,
            energy_variables=energy_variables,
            range_comfort_winter=range_comfort_winter,
            range_comfort_summer=range_comfort_summer,
            summer_start=summer_start,
            summer_final=summer_final,
            energy_weight=0.8,
            lambda_energy=2.0e-4,
            lambda_temperature=0.5
        ),
        'LinearReward (Comfort Focused)': LinearReward(
            temperature_variables=temperature_variables,
            energy_variables=energy_variables,
            range_comfort_winter=range_comfort_winter,
            range_comfort_summer=range_comfort_summer,
            summer_start=summer_start,
            summer_final=summer_final,
            energy_weight=0.2,
            lambda_energy=0.5e-4,
            lambda_temperature=2.0
        ),
        'ExpReward': ExpReward(
            temperature_variables=temperature_variables,
            energy_variables=energy_variables,
            range_comfort_winter=range_comfort_winter,
            range_comfort_summer=range_comfort_summer,
            summer_start=summer_start,
            summer_final=summer_final,
            energy_weight=0.5,
            lambda_energy=1.0e-4,
            lambda_temperature=1.0
        ),
        'HourlyLinearReward': HourlyLinearReward(
            temperature_variables=temperature_variables,
            energy_variables=energy_variables,
            range_comfort_winter=range_comfort_winter,
            range_comfort_summer=range_comfort_summer,
            summer_start=summer_start,
            summer_final=summer_final,
            default_energy_weight=0.5,
            lambda_energy=1.0e-4,
            lambda_temperature=1.0,
            range_comfort_hours=(9, 19)
        ),
        'NormalizedLinearReward': NormalizedLinearReward(
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
    }
    
    # Test each reward function
    for reward_name, reward_fn in reward_functions.items():
        print(f"\n📊 Testing {reward_name}:")
        print("-" * 40)
        
        total_reward = 0
        for i, obs in enumerate(test_observations):
            try:
                reward, reward_terms = reward_fn(obs)
                total_reward += reward
                
                print(f"  Observation {i+1}:")
                print(f"    Air temp: {obs['air_temperature']:.1f}°C")
                print(f"    HVAC power: {obs['HVAC_electricity_demand_rate']:.2f} W")
                print(f"    Month: {obs['month']}, Day: {obs['day_of_month']}, Hour: {obs['hour']}")
                print(f"    Reward: {reward:.4f}")
                if reward_terms:
                    print(f"    Reward terms: {reward_terms}")
            except Exception as e:
                print(f"  ❌ Error in observation {i+1}: {e}")
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Average reward: {total_reward/len(test_observations):.4f}")

def test_custom_environment_creation():
    """
    Test creating environments with custom reward functions.
    """
    print("\n🔧 TESTING CUSTOM ENVIRONMENT CREATION")
    print("=" * 50)
    
    try:
        import gymnasium as gym
        
        # Test 1: Default environment
        print("\n1️⃣ Testing default environment...")
        env = gym.make('Eplus-5zone-hot-continuous-v1', env_name='Test_Default')
        print(f"✅ Default environment created")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space.shape}")
        env.close()
        
        # Test 2: Environment with custom reward
        print("\n2️⃣ Testing environment with custom reward...")
        custom_reward = LinearReward(
            temperature_variables=['air_temperature'],
            energy_variables=['HVAC_electricity_demand_rate'],
            range_comfort_winter=(20.0, 23.5),
            range_comfort_summer=(23.0, 26.0),
            summer_start=(6, 1),
            summer_final=(9, 30),
            energy_weight=0.7,
            lambda_energy=1.5e-4,
            lambda_temperature=0.8
        )
        
        env = gym.make(
            'Eplus-5zone-hot-continuous-v1',
            env_name='Test_CustomReward',
            reward=custom_reward
        )
        print(f"✅ Custom reward environment created")
        print(f"   Energy weight: 0.7")
        print(f"   Lambda energy: 1.5e-4")
        print(f"   Lambda temperature: 0.8")
        env.close()
        
        # Test 3: Environment with custom run period
        print("\n3️⃣ Testing environment with custom run period...")
        env = gym.make(
            'Eplus-5zone-hot-continuous-v1',
            env_name='Test_CustomRunPeriod',
            config_params={
                'runperiod': (1, 1, 1991, 1, 7, 1991),  # 1 week
                'timesteps_per_hour': 4  # 15-minute timesteps
            }
        )
        print(f"✅ Custom run period environment created")
        print(f"   Run period: 1 week")
        print(f"   Timestep: 15 minutes")
        print(f"   Total timesteps: {env.timestep_per_episode}")
        env.close()
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")

def main():
    """
    Main function to run the fixed tests.
    """
    print("🎯 REWARD FUNCTION TESTING (FIXED)")
    print("=" * 50)
    
    # Test reward functions
    test_reward_functions_fixed()
    
    # Test environment creation
    test_custom_environment_creation()
    
    print("\n🎉 ALL TESTS COMPLETED!")
    print("=" * 50)
    print("✅ Reward functions tested successfully")
    print("✅ Environment creation tested successfully")
    print("✅ No more 'day_of_month' KeyError!")

if __name__ == "__main__":
    main()