#!/usr/bin/env python3
"""
5Zone Environment Customization Guide - Reward Weights & Run Periods
==================================================================

This comprehensive guide demonstrates how to customize:
1. Reward function weights (lambda_temp, lambda_energy, energy_weight)
2. Run periods (timestep size, episode length)
3. Verification methods to confirm changes are applied

Environment: Eplus-5zone-hot-continuous-v1

Key Customization Parameters:
- energy_weight: Controls balance between energy and comfort (0.0 to 1.0)
- lambda_energy: Scales energy consumption penalty
- lambda_temperature: Scales comfort violation penalty
- timestep_per_hour: Controls simulation timestep size (1, 2, 4, etc.)
- runperiod: Controls episode duration (start_month, start_day, end_month, end_day)
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Any

# =============================================================================
# STEP 1: IMPORTS AND SETUP
# =============================================================================

print("🚀 5Zone Environment Customization Guide")
print("=" * 60)

try:
    import gymnasium as gym
    import sinergym
    from sinergym.utils.rewards import LinearReward
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install sinergym[drl] stable-baselines3 gymnasium")
    sys.exit(1)

# =============================================================================
# STEP 2: REWARD WEIGHT CUSTOMIZATION
# =============================================================================

def demonstrate_reward_weights():
    """Demonstrate how to customize reward weights."""
    
    print("\n📊 REWARD WEIGHT CUSTOMIZATION")
    print("=" * 50)
    
    # Test observation: Hot summer day with high energy usage
    test_obs = {
        'air_temperature': 28.0,  # Too hot (above summer comfort range)
        'HVAC_electricity_demand_rate': 8000.0,  # High energy usage
        'month': 7, 'day_of_month': 15, 'hour': 14
    }
    
    print(f"🧪 Test Conditions:")
    print(f"   Temperature: {test_obs['air_temperature']}°C (Summer comfort: 23.0-26.0°C)")
    print(f"   Energy Usage: {test_obs['HVAC_electricity_demand_rate']}W")
    print(f"   Time: Month {test_obs['month']}, Day {test_obs['day_of_month']}, Hour {test_obs['hour']}")
    
    # Different reward configurations
    configs = [
        {
            'name': 'Default (Balanced)',
            'energy_weight': 0.5,
            'lambda_energy': 1e-4,
            'lambda_temperature': 1.0,
            'description': 'Equal weight to energy and comfort'
        },
        {
            'name': 'Energy-Focused',
            'energy_weight': 0.8,
            'lambda_energy': 2e-4,
            'lambda_temperature': 0.5,
            'description': 'Prioritizes energy savings (80% energy, 20% comfort)'
        },
        {
            'name': 'Comfort-Focused',
            'energy_weight': 0.2,
            'lambda_energy': 5e-5,
            'lambda_temperature': 2.0,
            'description': 'Prioritizes comfort (20% energy, 80% comfort)'
        },
        {
            'name': 'Extreme Energy Saving',
            'energy_weight': 0.9,
            'lambda_energy': 5e-4,
            'lambda_temperature': 0.1,
            'description': 'Maximum energy savings (90% energy, 10% comfort)'
        },
        {
            'name': 'Extreme Comfort Priority',
            'energy_weight': 0.1,
            'lambda_energy': 1e-5,
            'lambda_temperature': 5.0,
            'description': 'Maximum comfort priority (10% energy, 90% comfort)'
        }
    ]
    
    print(f"\n📊 Reward Calculation Results:")
    print("-" * 100)
    print(f"{'Configuration':<25} {'Energy Weight':<12} {'Lambda Energy':<15} {'Lambda Temp':<12} {'Reward':<10} {'Description'}")
    print("-" * 100)
    
    for config in configs:
        # Create reward function
        reward_fn = LinearReward(
            temperature_variables=['air_temperature'],
            energy_variables=['HVAC_electricity_demand_rate'],
            range_comfort_winter=(20.0, 23.5),
            range_comfort_summer=(23.0, 26.0),
            energy_weight=config['energy_weight'],
            lambda_energy=config['lambda_energy'],
            lambda_temperature=config['lambda_temperature']
        )
        
        # Calculate reward
        try:
            reward, terms = reward_fn(test_obs)
            print(f"{config['name']:<25} {config['energy_weight']:<12.2f} {config['lambda_energy']:<15.2e} {config['lambda_temperature']:<12.2f} {reward:<10.4f} {config['description']}")
        except Exception as e:
            print(f"{config['name']:<25} {'ERROR':<12} {'ERROR':<15} {'ERROR':<12} {'ERROR':<10} Error: {e}")
    
    print(f"\n💡 Key Insights:")
    print("• energy_weight controls the balance: 0.0 = all comfort, 1.0 = all energy")
    print("• lambda_energy scales the energy penalty (higher = stronger energy penalty)")
    print("• lambda_temperature scales the comfort penalty (higher = stronger comfort penalty)")
    print("• Lower rewards (more negative) indicate worse performance")

def create_custom_reward_function(energy_weight, lambda_energy, lambda_temperature):
    """Create a custom reward function with specified weights."""
    
    return LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(20.0, 23.5),
        range_comfort_summer=(23.0, 26.0),
        energy_weight=energy_weight,
        lambda_energy=lambda_energy,
        lambda_temperature=lambda_temperature
    )

# =============================================================================
# STEP 3: RUN PERIOD CUSTOMIZATION
# =============================================================================

def demonstrate_run_periods():
    """Demonstrate how to customize run periods."""
    
    print("\n⏱️ RUN PERIOD CUSTOMIZATION")
    print("=" * 50)
    
    # Different run period configurations
    run_periods = [
        {
            'name': 'Default (1 hour timesteps, 1 year)',
            'timestep_per_hour': 1,
            'runperiod': (1, 1, 12, 31),  # Jan 1 to Dec 31
            'timesteps_per_episode': 8760,
            'description': '1-hour timesteps for full year (8760 steps)'
        },
        {
            'name': '2-Hour Timesteps',
            'timestep_per_hour': 2,
            'runperiod': (1, 1, 12, 31),
            'timesteps_per_episode': 4380,  # 8760/2
            'description': '2-hour timesteps for full year (4380 steps)'
        },
        {
            'name': '4-Hour Timesteps',
            'timestep_per_hour': 4,
            'runperiod': (1, 1, 12, 31),
            'timesteps_per_episode': 2190,  # 8760/4
            'description': '4-hour timesteps for full year (2190 steps)'
        },
        {
            'name': 'Summer Only (June-August)',
            'timestep_per_hour': 1,
            'runperiod': (6, 1, 8, 31),  # June 1 to August 31
            'timesteps_per_episode': 2208,  # 92 days * 24 hours
            'description': '1-hour timesteps for summer months only'
        },
        {
            'name': 'Winter Only (December-February)',
            'timestep_per_hour': 1,
            'runperiod': (12, 1, 2, 28),  # December 1 to February 28
            'timesteps_per_episode': 2160,  # 90 days * 24 hours
            'description': '1-hour timesteps for winter months only'
        },
        {
            'name': 'Spring Only (March-May)',
            'timestep_per_hour': 1,
            'runperiod': (3, 1, 5, 31),  # March 1 to May 31
            'timesteps_per_episode': 2208,  # 92 days * 24 hours
            'description': '1-hour timesteps for spring months only'
        }
    ]
    
    print(f"📊 Run Period Configurations:")
    print("-" * 100)
    print(f"{'Configuration':<30} {'Timesteps/Hour':<15} {'Run Period':<15} {'Steps/Episode':<15} {'Description'}")
    print("-" * 100)
    
    for config in run_periods:
        hours_per_episode = config['timesteps_per_episode'] / config['timestep_per_hour']
        days_per_episode = hours_per_episode / 24
        
        print(f"{config['name']:<30} {config['timestep_per_hour']:<15} {str(config['runperiod']):<15} {config['timesteps_per_episode']:<15} {config['description']}")
        print(f"{'':<30} {'':<15} {'':<15} {'':<15}   Duration: {hours_per_episode:.0f} hours ({days_per_episode:.1f} days)")
    
    print(f"\n💡 Key Insights:")
    print("• timestep_per_hour: Controls simulation granularity (1=1hr, 2=2hr, etc.)")
    print("• runperiod: (start_month, start_day, end_month, end_day)")
    print("• timesteps_per_episode: Total simulation steps")
    print("• Shorter episodes = Faster training, less comprehensive")
    print("• Longer timesteps = Less granular control, faster simulation")

# =============================================================================
# STEP 4: ENVIRONMENT CREATION WITH CUSTOM CONFIGURATIONS
# =============================================================================

def create_custom_environment(reward_config=None, run_period_config=None):
    """Create environment with custom reward and run period configurations."""
    
    print(f"\n🔧 CREATING CUSTOM ENVIRONMENT")
    print("=" * 50)
    
    try:
        # Create base environment
        env = gym.make('Eplus-5zone-hot-continuous-v1')
        
        # Apply custom reward function if provided
        if reward_config:
            print(f"📊 Applying custom reward configuration:")
            print(f"   Energy Weight: {reward_config['energy_weight']:.2f}")
            print(f"   Lambda Energy: {reward_config['lambda_energy']:.2e}")
            print(f"   Lambda Temperature: {reward_config['lambda_temperature']:.2f}")
            
            custom_reward = LinearReward(
                temperature_variables=['air_temperature'],
                energy_variables=['HVAC_electricity_demand_rate'],
                range_comfort_winter=(20.0, 23.5),
                range_comfort_summer=(23.0, 26.0),
                energy_weight=reward_config['energy_weight'],
                lambda_energy=reward_config['lambda_energy'],
                lambda_temperature=reward_config['lambda_temperature']
            )
            env.reward_fn = custom_reward
        
        # Apply custom run period if provided
        if run_period_config:
            print(f"⏱️ Applying custom run period configuration:")
            print(f"   Timesteps per Hour: {run_period_config['timestep_per_hour']}")
            print(f"   Run Period: {run_period_config['runperiod']}")
            print(f"   Timesteps per Episode: {run_period_config['timesteps_per_episode']}")
            
            env.timestep_per_hour = run_period_config['timestep_per_hour']
            env.runperiod = run_period_config['runperiod']
        
        print("✅ Environment created successfully")
        return env
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return None

def verify_environment_configuration(env, reward_config=None, run_period_config=None):
    """Verify that environment has correct configurations."""
    
    print(f"\n🔍 VERIFYING ENVIRONMENT CONFIGURATION")
    print("=" * 50)
    
    # Verify run period settings
    if run_period_config:
        print(f"Timestep per Hour: {env.timestep_per_hour} (Expected: {run_period_config['timestep_per_hour']})")
        print(f"Run Period: {env.runperiod} (Expected: {run_period_config['runperiod']})")
        print(f"Timesteps per Episode: {env.timestep_per_episode:,} (Expected: {run_period_config['timesteps_per_episode']:,})")
    
    # Verify reward function
    if reward_config and hasattr(env, 'reward_fn') and env.reward_fn is not None:
        print(f"Reward Function: {type(env.reward_fn).__name__}")
        
        # Check reward parameters
        if hasattr(env.reward_fn, 'energy_weight'):
            print(f"Energy Weight: {env.reward_fn.energy_weight:.2f} (Expected: {reward_config['energy_weight']:.2f})")
        if hasattr(env.reward_fn, 'lambda_energy'):
            print(f"Lambda Energy: {env.reward_fn.lambda_energy:.2e} (Expected: {reward_config['lambda_energy']:.2e})")
        if hasattr(env.reward_fn, 'lambda_temperature'):
            print(f"Lambda Temperature: {env.reward_fn.lambda_temperature:.2f} (Expected: {reward_config['lambda_temperature']:.2f})")
    else:
        print("Using default reward function")

# =============================================================================
# STEP 5: PRACTICAL EXAMPLES
# =============================================================================

def run_practical_examples():
    """Run practical examples of customization."""
    
    print("\n🎬 PRACTICAL CUSTOMIZATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Energy-focused configuration with 2-hour timesteps
    print(f"\n📋 Example 1: Energy-Focused Configuration with 2-Hour Timesteps")
    print("-" * 60)
    
    reward_config_1 = {
        'energy_weight': 0.8,
        'lambda_energy': 2e-4,
        'lambda_temperature': 0.5
    }
    
    run_period_config_1 = {
        'timestep_per_hour': 2,
        'runperiod': (1, 1, 12, 31),
        'timesteps_per_episode': 4380
    }
    
    env_1 = create_custom_environment(reward_config_1, run_period_config_1)
    if env_1:
        verify_environment_configuration(env_1, reward_config_1, run_period_config_1)
        env_1.close()
    
    # Example 2: Comfort-focused configuration with summer-only period
    print(f"\n📋 Example 2: Comfort-Focused Configuration with Summer-Only Period")
    print("-" * 60)
    
    reward_config_2 = {
        'energy_weight': 0.2,
        'lambda_energy': 5e-5,
        'lambda_temperature': 2.0
    }
    
    run_period_config_2 = {
        'timestep_per_hour': 1,
        'runperiod': (6, 1, 8, 31),
        'timesteps_per_episode': 2208
    }
    
    env_2 = create_custom_environment(reward_config_2, run_period_config_2)
    if env_2:
        verify_environment_configuration(env_2, reward_config_2, run_period_config_2)
        env_2.close()
    
    # Example 3: Balanced configuration with 4-hour timesteps
    print(f"\n📋 Example 3: Balanced Configuration with 4-Hour Timesteps")
    print("-" * 60)
    
    reward_config_3 = {
        'energy_weight': 0.5,
        'lambda_energy': 1e-4,
        'lambda_temperature': 1.0
    }
    
    run_period_config_3 = {
        'timestep_per_hour': 4,
        'runperiod': (1, 1, 12, 31),
        'timesteps_per_episode': 2190
    }
    
    env_3 = create_custom_environment(reward_config_3, run_period_config_3)
    if env_3:
        verify_environment_configuration(env_3, reward_config_3, run_period_config_3)
        env_3.close()

# =============================================================================
# STEP 6: USAGE PATTERNS AND BEST PRACTICES
# =============================================================================

def show_usage_patterns():
    """Show common usage patterns and best practices."""
    
    print("\n📚 USAGE PATTERNS AND BEST PRACTICES")
    print("=" * 60)
    
    print(f"\n🎯 Common Use Cases:")
    
    print(f"\n1. Energy Optimization Research:")
    print("   reward_config = {")
    print("       'energy_weight': 0.8,")
    print("       'lambda_energy': 2e-4,")
    print("       'lambda_temperature': 0.5")
    print("   }")
    print("   run_period_config = {")
    print("       'timestep_per_hour': 1,")
    print("       'runperiod': (1, 1, 12, 31)")
    print("   }")
    
    print(f"\n2. Comfort Optimization Research:")
    print("   reward_config = {")
    print("       'energy_weight': 0.2,")
    print("       'lambda_energy': 5e-5,")
    print("       'lambda_temperature': 2.0")
    print("   }")
    print("   run_period_config = {")
    print("       'timestep_per_hour': 1,")
    print("       'runperiod': (6, 1, 8, 31)  # Summer only")
    print("   }")
    
    print(f"\n3. Fast Training (Coarse Timesteps):")
    print("   reward_config = {")
    print("       'energy_weight': 0.5,")
    print("       'lambda_energy': 1e-4,")
    print("       'lambda_temperature': 1.0")
    print("   }")
    print("   run_period_config = {")
    print("       'timestep_per_hour': 4,")
    print("       'runperiod': (1, 1, 12, 31)")
    print("   }")
    
    print(f"\n4. Seasonal Analysis:")
    print("   # Winter analysis")
    print("   run_period_config = {")
    print("       'timestep_per_hour': 1,")
    print("       'runperiod': (12, 1, 2, 28)")
    print("   }")
    print("   # Summer analysis")
    print("   run_period_config = {")
    print("       'timestep_per_hour': 1,")
    print("       'runperiod': (6, 1, 8, 31)")
    print("   }")
    
    print(f"\n💡 Best Practices:")
    print("• Start with balanced energy_weight (0.5) and adjust based on priorities")
    print("• Use 1-hour timesteps for detailed analysis, 2-4 hours for faster training")
    print("• Test different lambda values to find optimal scaling")
    print("• Consider seasonal periods for focused analysis")
    print("• Always verify configurations before training")
    print("• Monitor both energy consumption and comfort violations")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the comprehensive customization guide."""
    
    print("🚀 Starting 5Zone Environment Customization Guide")
    print("This guide demonstrates reward weight and run period customization")
    print("=" * 80)
    
    try:
        # Demonstrate reward weight customization
        demonstrate_reward_weights()
        
        # Demonstrate run period customization
        demonstrate_run_periods()
        
        # Run practical examples
        run_practical_examples()
        
        # Show usage patterns
        show_usage_patterns()
        
        print("\n✅ Guide completed successfully!")
        print("\n🎯 Key Takeaways:")
        print("• energy_weight: Controls balance between energy and comfort (0.0-1.0)")
        print("• lambda_energy: Scales energy consumption penalty")
        print("• lambda_temperature: Scales comfort violation penalty")
        print("• timestep_per_hour: Controls simulation granularity")
        print("• runperiod: Controls episode duration and season")
        print("• All changes can be verified through environment inspection")
        
        print(f"\n📖 Next Steps:")
        print("1. Experiment with different reward configurations")
        print("2. Test various run periods for your specific use case")
        print("3. Monitor training performance with different settings")
        print("4. Consider seasonal analysis for climate-specific optimization")
        
    except KeyboardInterrupt:
        print("\n⏹️ Guide interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during guide: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()