#!/usr/bin/env python3
"""
5Zone Environment - Reward Weights & Run Period Customization
============================================================

This script demonstrates how to customize:
1. Reward function weights (lambda_temp, lambda_energy, energy_weight)
2. Run periods (timestep size, episode length)
3. Verification methods to confirm changes are applied

Environment: Eplus-5zone-hot-continuous-v1
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# =============================================================================
# STEP 1: IMPORTS AND SETUP
# =============================================================================

print("🚀 5Zone Environment Customization Demo")
print("=" * 60)

try:
    import gymnasium as gym
    import sinergym
    from sinergym.utils.rewards import LinearReward
    from sinergym.utils.wrappers import LoggerWrapper
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install sinergym[drl] stable-baselines3 gymnasium")
    sys.exit(1)

# =============================================================================
# STEP 2: CUSTOM REWARD CONFIGURATIONS
# =============================================================================

def create_custom_reward_configs():
    """Create different reward configurations for testing."""
    
    configs = {
        'default': {
            'name': 'Default Configuration',
            'energy_weight': 0.5,
            'lambda_energy': 1e-4,
            'lambda_temperature': 1.0,
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'energy_focused': {
            'name': 'Energy-Focused Configuration',
            'energy_weight': 0.8,  # Higher weight on energy
            'lambda_energy': 2e-4,  # Higher energy penalty
            'lambda_temperature': 0.5,  # Lower comfort penalty
            'range_comfort_winter': (18.0, 25.0),  # Wider comfort range
            'range_comfort_summer': (22.0, 28.0)
        },
        'comfort_focused': {
            'name': 'Comfort-Focused Configuration',
            'energy_weight': 0.2,  # Lower weight on energy
            'lambda_energy': 5e-5,  # Lower energy penalty
            'lambda_temperature': 2.0,  # Higher comfort penalty
            'range_comfort_winter': (20.5, 22.5),  # Narrower comfort range
            'range_comfort_summer': (23.5, 25.5)
        },
        'balanced': {
            'name': 'Balanced Configuration',
            'energy_weight': 0.5,
            'lambda_energy': 1e-4,
            'lambda_temperature': 1.0,
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        }
    }
    
    return configs

def print_reward_config(config: Dict, config_name: str):
    """Print reward configuration details."""
    print(f"\n📊 {config_name.upper()} REWARD CONFIGURATION")
    print("-" * 50)
    print(f"Energy Weight: {config['energy_weight']:.2f}")
    print(f"Lambda Energy: {config['lambda_energy']:.2e}")
    print(f"Lambda Temperature: {config['lambda_temperature']:.2f}")
    print(f"Winter Comfort Range: {config['range_comfort_winter']} °C")
    print(f"Summer Comfort Range: {config['range_comfort_summer']} °C")
    
    # Calculate comfort term weight
    comfort_weight = 1 - config['energy_weight']
    print(f"Comfort Weight: {comfort_weight:.2f}")
    
    # Show formula
    print(f"\nReward Formula:")
    print(f"R = -{config['energy_weight']:.2f} × {config['lambda_energy']:.2e} × energy")
    print(f"    -{comfort_weight:.2f} × {config['lambda_temperature']:.2f} × temp_violation")

# =============================================================================
# STEP 3: RUN PERIOD CONFIGURATIONS
# =============================================================================

def create_run_period_configs():
    """Create different run period configurations."""
    
    configs = {
        'default': {
            'name': 'Default (1 hour timesteps, 1 year)',
            'timestep_per_hour': 1,
            'runperiod': (1, 1, 12, 31),  # (start_month, start_day, end_month, end_day)
            'timesteps_per_episode': 8760,
            'description': '1-hour timesteps for full year'
        },
        'two_hour': {
            'name': '2-Hour Timesteps',
            'timestep_per_hour': 2,
            'runperiod': (1, 1, 12, 31),
            'timesteps_per_episode': 4380,  # 8760/2
            'description': '2-hour timesteps for full year'
        },
        'four_hour': {
            'name': '4-Hour Timesteps',
            'timestep_per_hour': 4,
            'runperiod': (1, 1, 12, 31),
            'timesteps_per_episode': 2190,  # 8760/4
            'description': '4-hour timesteps for full year'
        },
        'summer_only': {
            'name': 'Summer Only (June-August)',
            'timestep_per_hour': 1,
            'runperiod': (6, 1, 8, 31),  # June 1 to August 31
            'timesteps_per_episode': 2208,  # 92 days * 24 hours
            'description': '1-hour timesteps for summer months only'
        },
        'winter_only': {
            'name': 'Winter Only (December-February)',
            'timestep_per_hour': 1,
            'runperiod': (12, 1, 2, 28),  # December 1 to February 28
            'timesteps_per_episode': 2160,  # 90 days * 24 hours
            'description': '1-hour timesteps for winter months only'
        }
    }
    
    return configs

def print_run_period_config(config: Dict, config_name: str):
    """Print run period configuration details."""
    print(f"\n⏱️ {config_name.upper()} RUN PERIOD CONFIGURATION")
    print("-" * 50)
    print(f"Timesteps per Hour: {config['timestep_per_hour']}")
    print(f"Run Period: {config['runperiod']}")
    print(f"Timesteps per Episode: {config['timesteps_per_episode']:,}")
    print(f"Description: {config['description']}")
    
    # Calculate episode duration
    hours_per_episode = config['timesteps_per_episode'] / config['timestep_per_hour']
    days_per_episode = hours_per_episode / 24
    print(f"Episode Duration: {hours_per_episode:.0f} hours ({days_per_episode:.1f} days)")

# =============================================================================
# STEP 4: ENVIRONMENT CREATION WITH CUSTOM CONFIGURATIONS
# =============================================================================

def create_custom_environment(reward_config: Dict, run_period_config: Dict):
    """Create environment with custom reward and run period configurations."""
    
    # Create environment configuration
    env_config = {
        'id': 'Eplus-5zone-hot-continuous-v1',
        'timestep_per_hour': run_period_config['timestep_per_hour'],
        'runperiod': run_period_config['runperiod'],
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': reward_config['range_comfort_winter'],
            'range_comfort_summer': reward_config['range_comfort_summer'],
            'energy_weight': reward_config['energy_weight'],
            'lambda_energy': reward_config['lambda_energy'],
            'lambda_temperature': reward_config['lambda_temperature']
        }
    }
    
    print(f"\n🔧 Creating environment with:")
    print(f"   Reward: {reward_config['name']}")
    print(f"   Run Period: {run_period_config['name']}")
    
    try:
        # Create environment
        env = gym.make(env_config['id'])
        
        # Apply custom configurations
        env.timestep_per_hour = env_config['timestep_per_hour']
        env.runperiod = env_config['runperiod']
        
        # Create and set custom reward function
        custom_reward = LinearReward(**env_config['reward_kwargs'])
        env.reward_fn = custom_reward
        
        print("✅ Environment created successfully")
        return env
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return None

# =============================================================================
# STEP 5: VERIFICATION AND TESTING FUNCTIONS
# =============================================================================

def verify_environment_configuration(env, reward_config: Dict, run_period_config: Dict):
    """Verify that environment has correct configurations."""
    
    print(f"\n🔍 VERIFYING ENVIRONMENT CONFIGURATION")
    print("=" * 50)
    
    # Verify run period settings
    print(f"Timestep per Hour: {env.timestep_per_hour} (Expected: {run_period_config['timestep_per_hour']})")
    print(f"Run Period: {env.runperiod} (Expected: {run_period_config['runperiod']})")
    print(f"Timesteps per Episode: {env.timestep_per_episode:,} (Expected: {run_period_config['timesteps_per_episode']:,})")
    
    # Verify reward function
    if hasattr(env, 'reward_fn') and env.reward_fn is not None:
        print(f"Reward Function: {type(env.reward_fn).__name__}")
        
        # Check reward parameters
        if hasattr(env.reward_fn, 'energy_weight'):
            print(f"Energy Weight: {env.reward_fn.energy_weight:.2f} (Expected: {reward_config['energy_weight']:.2f})")
        if hasattr(env.reward_fn, 'lambda_energy'):
            print(f"Lambda Energy: {env.reward_fn.lambda_energy:.2e} (Expected: {reward_config['lambda_energy']:.2e})")
        if hasattr(env.reward_fn, 'lambda_temperature'):
            print(f"Lambda Temperature: {env.reward_fn.lambda_temperature:.2f} (Expected: {reward_config['lambda_temperature']:.2f})")
    else:
        print("❌ No custom reward function found")

def test_reward_calculation(env, test_observations: List[Dict]):
    """Test reward calculation with different observations."""
    
    print(f"\n🧪 TESTING REWARD CALCULATION")
    print("=" * 50)
    
    for i, obs_dict in enumerate(test_observations):
        print(f"\nTest Case {i+1}:")
        print(f"  Temperature: {obs_dict['air_temperature']}°C")
        print(f"  Energy: {obs_dict['HVAC_electricity_demand_rate']}W")
        print(f"  Month: {obs_dict['month']}, Day: {obs_dict['day_of_month']}, Hour: {obs_dict['hour']}")
        
        # Convert dict to observation array
        obs_array = np.array([
            obs_dict['month'], obs_dict['day_of_month'], obs_dict['hour'],
            25.0, 50.0, 2.0, 180.0, 100.0, 500.0,  # Weather variables
            20.0, 24.0,  # Setpoints
            obs_dict['air_temperature'], 45.0, 10.0, 15.0,  # Zone variables
            obs_dict['HVAC_electricity_demand_rate'],  # Energy
            1000000.0  # Total electricity
        ])
        
        try:
            # Calculate reward
            reward, reward_terms = env.reward_fn(obs_dict)
            print(f"  Reward: {reward:.4f}")
            
            # Show reward components if available
            if isinstance(reward_terms, dict):
                for term, value in reward_terms.items():
                    print(f"    {term}: {value:.4f}")
                    
        except Exception as e:
            print(f"  ❌ Error calculating reward: {e}")

def run_episode_demo(env, max_steps: int = 100):
    """Run a short episode to demonstrate the environment."""
    
    print(f"\n🎮 RUNNING EPISODE DEMO ({max_steps} steps)")
    print("=" * 50)
    
    try:
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        
        total_reward = 0
        energy_consumption = []
        temperature_violations = []
        
        for step in range(min(max_steps, env.timestep_per_episode)):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Collect metrics
            if 'total_power_demand' in info:
                energy_consumption.append(info['total_power_demand'])
            if 'total_temperature_violation' in info:
                temperature_violations.append(info['total_temperature_violation'])
            
            if step % 20 == 0:  # Print every 20 steps
                print(f"Step {step:3d}: Reward = {reward:8.4f}, Total = {total_reward:8.4f}")
            
            if terminated or truncated:
                break
        
        print(f"\nEpisode Summary:")
        print(f"  Total Steps: {step + 1}")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Average Reward: {total_reward / (step + 1):.4f}")
        
        if energy_consumption:
            print(f"  Average Energy: {np.mean(energy_consumption):.2f}W")
        if temperature_violations:
            print(f"  Average Temp Violation: {np.mean(temperature_violations):.4f}°C")
            
    except Exception as e:
        print(f"❌ Error during episode: {e}")

# =============================================================================
# STEP 6: MAIN DEMONSTRATION FUNCTION
# =============================================================================

def run_comprehensive_demo():
    """Run comprehensive demonstration of all customizations."""
    
    print("\n🎬 COMPREHENSIVE CUSTOMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Get configurations
    reward_configs = create_custom_reward_configs()
    run_period_configs = create_run_period_configs()
    
    # Test observations
    test_observations = [
        {
            'air_temperature': 22.0,  # Comfortable
            'HVAC_electricity_demand_rate': 5000.0,  # Moderate energy
            'month': 1, 'day_of_month': 15, 'hour': 12
        },
        {
            'air_temperature': 28.0,  # Too hot
            'HVAC_electricity_demand_rate': 8000.0,  # High energy
            'month': 7, 'day_of_month': 15, 'hour': 14
        },
        {
            'air_temperature': 18.0,  # Too cold
            'HVAC_electricity_demand_rate': 6000.0,  # High energy
            'month': 12, 'day_of_month': 15, 'hour': 8
        }
    ]
    
    # Test different configurations
    test_cases = [
        ('default', 'default'),
        ('energy_focused', 'default'),
        ('comfort_focused', 'default'),
        ('default', 'two_hour'),
        ('default', 'summer_only')
    ]
    
    for reward_key, run_period_key in test_cases:
        print(f"\n{'='*60}")
        print(f"TESTING: {reward_configs[reward_key]['name']} + {run_period_configs[run_period_key]['name']}")
        print(f"{'='*60}")
        
        # Print configurations
        print_reward_config(reward_configs[reward_key], reward_key)
        print_run_period_config(run_period_configs[run_period_key], run_period_key)
        
        # Create environment
        env = create_custom_environment(reward_configs[reward_key], run_period_configs[run_period_key])
        
        if env is not None:
            # Verify configuration
            verify_environment_configuration(env, reward_configs[reward_key], run_period_configs[run_period_key])
            
            # Test reward calculation
            test_reward_calculation(env, test_observations)
            
            # Run short episode
            run_episode_demo(env, max_steps=50)
            
            # Close environment
            env.close()
        
        print(f"\n✅ Test case completed: {reward_key} + {run_period_key}")

# =============================================================================
# STEP 7: INTERACTIVE CONFIGURATION TESTER
# =============================================================================

def interactive_configuration_tester():
    """Interactive tool to test different configurations."""
    
    print("\n🎛️ INTERACTIVE CONFIGURATION TESTER")
    print("=" * 50)
    
    reward_configs = create_custom_reward_configs()
    run_period_configs = create_run_period_configs()
    
    print("\nAvailable Reward Configurations:")
    for i, (key, config) in enumerate(reward_configs.items(), 1):
        print(f"{i}. {key}: {config['name']}")
    
    print("\nAvailable Run Period Configurations:")
    for i, (key, config) in enumerate(run_period_configs.items(), 1):
        print(f"{i}. {key}: {config['name']}")
    
    # Let user select configurations
    try:
        reward_choice = input("\nSelect reward configuration (1-4): ")
        run_period_choice = input("Select run period configuration (1-5): ")
        
        reward_keys = list(reward_configs.keys())
        run_period_keys = list(run_period_configs.keys())
        
        reward_key = reward_keys[int(reward_choice) - 1]
        run_period_key = run_period_keys[int(run_period_choice) - 1]
        
        print(f"\nSelected: {reward_configs[reward_key]['name']} + {run_period_configs[run_period_key]['name']}")
        
        # Create and test environment
        env = create_custom_environment(reward_configs[reward_key], run_period_configs[run_period_key])
        
        if env is not None:
            verify_environment_configuration(env, reward_configs[reward_key], run_period_configs[run_period_key])
            test_reward_calculation(env, [
                {'air_temperature': 25.0, 'HVAC_electricity_demand_rate': 6000.0, 'month': 7, 'day_of_month': 15, 'hour': 12}
            ])
            env.close()
            
    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the demonstration."""
    
    print("🚀 Starting 5Zone Environment Customization Demo")
    print("This demo shows how to customize reward weights and run periods")
    print("=" * 80)
    
    try:
        # Run comprehensive demonstration
        run_comprehensive_demo()
        
        # Run interactive tester
        interactive_configuration_tester()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey Takeaways:")
        print("• Reward weights can be customized through LinearReward parameters")
        print("• Run periods can be modified using timestep_per_hour and runperiod")
        print("• All changes are verified through environment inspection")
        print("• Test observations help validate reward calculations")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()