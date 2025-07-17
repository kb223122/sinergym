#!/usr/bin/env python3
Test script to verify the fixed Sinergym environment creation
"""
"""

import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import (
    LoggerWrapper, CSVLogger, NormalizeObservation, NormalizeAction
)
from sinergym.utils.rewards import LinearReward

def test_environment_creation():
    """Test that environment creation works without the config_params error   
    print(=== Testing Fixed Environment Creation ===\n)
    
    # Test 1: Basic environment creation
    print("1. Testing basic environment creation...")
    try:
        env = gym.make('Eplus-5zone-hot-continuous-v1)
        print("   ✓ Basic environment created successfully)
        print(f"   Environment name: {env.name})
        print(f"   Observation space: {env.observation_space})
        print(f"   Action space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"   ✗ Error creating basic environment: {e}")
        return False
    
    # Test 2: Environment with wrappers
    print("\n2. Testing environment with wrappers...")
    try:
        env = gym.make('Eplus-5zone-hot-continuous-v1')
        env = LoggerWrapper(env)
        env = CSVLogger(env)
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        print("   ✓ Environment with wrappers created successfully")
        env.close()
    except Exception as e:
        print(f"   ✗ Error creating environment with wrappers: {e}")
        return False
    
    # Test 3: Environment with custom reward
    print("\n3. Testing environment with custom reward...")
    try:
        reward_kwargs = {
        temperature_variables": ["air_temperature"],
           energy_variables:["HVAC_electricity_demand_rate"],
          range_comfort_winter": [20.0, 230.5            range_comfort_summer": [23.0, 260,
            summer_start": [61,
            summer_final": [9, 30           energy_weight": 0.4           lambda_energy":00.01  lambda_temperature": 28
        }
        
        env = gym.make('Eplus-5zone-hot-continuous-v1')
        env = LoggerWrapper(env)
        env = CSVLogger(env)
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        env.set_wrapper_attr('reward_fn, LinearReward(**reward_kwargs))
        print("   ✓ Environment with custom reward created successfully")
        env.close()
    except Exception as e:
        print(f"   ✗ Error creating environment with custom reward: {e}")
        return False
    
    # Test4Simple episode run
    print("\n4. Testing simple episode run...")
    try:
        env = gym.make('Eplus-5zone-hot-continuous-v1')
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        
        obs, info = env.reset()
        print(f"   ✓ Environment reset successful)
        print(f"   Initial observation shape: {obs.shape}")
        
        # Run a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: Reward = {reward:0.3}")
            if terminated or truncated:
                break
        
        env.close()
        print("   ✓ Episode run completed successfully")
    except Exception as e:
        print(f   ✗ Error during episode run: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("The environment creation fix is working correctly.)
    return True

def test_original_error():
   that the original error still occurs with the problematic code
    
    print("\n=== Testing Original Error (Expected to Fail) ===\n")
    
    try:
        # This should fail with the original error
        extra_conf = [object Object]         timesteps_per_hour': 1,
            runperiod: (1,1 19911991     reward[object Object]
            temperature_variables': ['air_temperature'],
               energy_variables:['HVAC_electricity_demand_rate'],
              range_comfort_winter': [20.0, 23.5],
                range_comfort_summer': [23.0, 26.0],
                summer_start': [6, 1],
                summer_final': [9, 30],
                energy_weight': 0.4
                lambda_energy': 0.01
       lambda_temperature':28         }
        }
        
        env = gym.make('Eplus-5zone-hot-continuous-v1', 
                      config_params=extra_conf, 
                      env_name='test-env)
        print("   ✗ This should have failed but didn't!")
        env.close()
        return False
    except TypeError as e:
        if config_params" in str(e):
            print("   ✓ Original error correctly reproduced")
            print(f  Error message: {e}")
            return True
        else:
            print(f"   ✗ Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Sinergym Environment Fix Test)
    print(= * 50   
    # Test the fixed approach
    success1 = test_environment_creation()
    
    # Test that the original error still occurs
    success2 = test_original_error()
    
    if success1 and success2:
        print("\n" +=50)
        print("✓ ALL TESTS PASSED!)        print("The fix is working correctly.")
        print("\nTo use the fixed approach in your code:)
        print("1. Remove config_params and env_name from gym.make())
        print("2. Use set_wrapper_attr() to set custom reward functions)
        print("3. Use pre-configured environments when possible")
    else:
        print("\n" +=50)
        print(✗ SOME TESTS FAILED!)     print("Please check the error messages above.")