#!/usr/bin/env python3
"""
Simple Reward Function Testing - No EnergyPlus Required
======================================================

This script demonstrates how to customize reward weights in Sinergym
without requiring EnergyPlus installation. It focuses on:
1. Testing different reward configurations
2. Verifying lambda_temp, lambda_energy, and energy_weight changes
3. Comparing reward calculations across configurations

Environment: Eplus-5zone-hot-continuous-v1 (reward testing only)
"""

import numpy as np
from typing import Dict, List

# =============================================================================
# STEP 1: IMPORTS AND SETUP
# =============================================================================

print("🚀 Simple Reward Function Testing")
print("=" * 50)

try:
    from sinergym.utils.rewards import LinearReward
    print("✅ Sinergym rewards imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install sinergym[drl]")
    exit(1)

# =============================================================================
# STEP 2: REWARD CONFIGURATIONS
# =============================================================================

def create_reward_configurations():
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
            'name': 'Energy-Focused (High Energy Weight)',
            'energy_weight': 0.8,  # 80% weight on energy
            'lambda_energy': 2e-4,  # Higher energy penalty
            'lambda_temperature': 0.5,  # Lower comfort penalty
            'range_comfort_winter': (18.0, 25.0),  # Wider comfort range
            'range_comfort_summer': (22.0, 28.0)
        },
        'comfort_focused': {
            'name': 'Comfort-Focused (Low Energy Weight)',
            'energy_weight': 0.2,  # 20% weight on energy
            'lambda_energy': 5e-5,  # Lower energy penalty
            'lambda_temperature': 2.0,  # Higher comfort penalty
            'range_comfort_winter': (20.5, 22.5),  # Narrower comfort range
            'range_comfort_summer': (23.5, 25.5)
        },
        'extreme_energy': {
            'name': 'Extreme Energy Saving',
            'energy_weight': 0.9,  # 90% weight on energy
            'lambda_energy': 5e-4,  # Very high energy penalty
            'lambda_temperature': 0.1,  # Very low comfort penalty
            'range_comfort_winter': (15.0, 30.0),  # Very wide comfort range
            'range_comfort_summer': (20.0, 35.0)
        },
        'extreme_comfort': {
            'name': 'Extreme Comfort Priority',
            'energy_weight': 0.1,  # 10% weight on energy
            'lambda_energy': 1e-5,  # Very low energy penalty
            'lambda_temperature': 5.0,  # Very high comfort penalty
            'range_comfort_winter': (21.0, 22.0),  # Very narrow comfort range
            'range_comfort_summer': (24.0, 25.0)
        }
    }
    
    return configs

def print_reward_config(config: Dict, config_name: str):
    """Print detailed reward configuration."""
    print(f"\n📊 {config_name.upper()} REWARD CONFIGURATION")
    print("-" * 60)
    print(f"Energy Weight: {config['energy_weight']:.2f} ({config['energy_weight']*100:.0f}%)")
    print(f"Lambda Energy: {config['lambda_energy']:.2e}")
    print(f"Lambda Temperature: {config['lambda_temperature']:.2f}")
    print(f"Winter Comfort Range: {config['range_comfort_winter']} °C")
    print(f"Summer Comfort Range: {config['range_comfort_summer']} °C")
    
    # Calculate comfort term weight
    comfort_weight = 1 - config['energy_weight']
    print(f"Comfort Weight: {comfort_weight:.2f} ({comfort_weight*100:.0f}%)")
    
    # Show formula
    print(f"\nReward Formula:")
    print(f"R = -{config['energy_weight']:.2f} × {config['lambda_energy']:.2e} × energy")
    print(f"    -{comfort_weight:.2f} × {config['lambda_temperature']:.2f} × temp_violation")
    
    # Calculate relative penalties
    energy_penalty = config['energy_weight'] * config['lambda_energy']
    comfort_penalty = comfort_weight * config['lambda_temperature']
    print(f"\nRelative Penalties:")
    print(f"Energy Penalty Coefficient: {energy_penalty:.2e}")
    print(f"Comfort Penalty Coefficient: {comfort_penalty:.2f}")
    print(f"Comfort/Energy Ratio: {comfort_penalty/energy_penalty:.2e}")

# =============================================================================
# STEP 3: TEST OBSERVATIONS
# =============================================================================

def create_test_observations():
    """Create test observations for reward calculation."""
    
    observations = [
        {
            'name': 'Comfortable Winter',
            'air_temperature': 22.0,  # Within winter comfort range
            'HVAC_electricity_demand_rate': 3000.0,  # Low energy
            'month': 1, 'day_of_month': 15, 'hour': 12
        },
        {
            'name': 'Comfortable Summer',
            'air_temperature': 24.5,  # Within summer comfort range
            'HVAC_electricity_demand_rate': 4000.0,  # Moderate energy
            'month': 7, 'day_of_month': 15, 'hour': 14
        },
        {
            'name': 'Too Hot (Summer)',
            'air_temperature': 28.0,  # Above summer comfort range
            'HVAC_electricity_demand_rate': 8000.0,  # High energy
            'month': 7, 'day_of_month': 15, 'hour': 14
        },
        {
            'name': 'Too Cold (Winter)',
            'air_temperature': 18.0,  # Below winter comfort range
            'HVAC_electricity_demand_rate': 6000.0,  # High energy
            'month': 1, 'day_of_month': 15, 'hour': 8
        },
        {
            'name': 'Extreme Energy Usage',
            'air_temperature': 25.0,  # Comfortable
            'HVAC_electricity_demand_rate': 15000.0,  # Very high energy
            'month': 7, 'day_of_month': 15, 'hour': 16
        },
        {
            'name': 'Extreme Temperature Violation',
            'air_temperature': 35.0,  # Very hot
            'HVAC_electricity_demand_rate': 5000.0,  # Moderate energy
            'month': 7, 'day_of_month': 15, 'hour': 14
        }
    ]
    
    return observations

# =============================================================================
# STEP 4: REWARD CALCULATION TESTING
# =============================================================================

def create_reward_function(config: Dict):
    """Create a reward function with the given configuration."""
    
    reward_kwargs = {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': config['range_comfort_winter'],
        'range_comfort_summer': config['range_comfort_summer'],
        'energy_weight': config['energy_weight'],
        'lambda_energy': config['lambda_energy'],
        'lambda_temperature': config['lambda_temperature']
    }
    
    return LinearReward(**reward_kwargs)

def test_reward_calculation(reward_fn, obs_dict: Dict):
    """Test reward calculation for a single observation."""
    
    try:
        # Calculate reward
        reward, reward_terms = reward_fn(obs_dict)
        
        # Extract components
        energy_term = reward_terms.get('energy_term', 0)
        temperature_term = reward_terms.get('temperature_term', 0)
        total_reward = reward
        
        return {
            'total_reward': total_reward,
            'energy_term': energy_term,
            'temperature_term': temperature_term,
            'success': True
        }
        
    except Exception as e:
        return {
            'total_reward': 0,
            'energy_term': 0,
            'temperature_term': 0,
            'success': False,
            'error': str(e)
        }

def compare_reward_configurations():
    """Compare reward calculations across different configurations."""
    
    print("\n🎬 COMPREHENSIVE REWARD CONFIGURATION COMPARISON")
    print("=" * 80)
    
    # Get configurations and test observations
    configs = create_reward_configurations()
    test_obs = create_test_observations()
    
    # Create results table
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"TESTING: {config['name']}")
        print(f"{'='*80}")
        
        # Print configuration
        print_reward_config(config, config_name)
        
        # Create reward function
        reward_fn = create_reward_function(config)
        print(f"\n✅ Reward function created successfully")
        
        # Test all observations
        config_results = {}
        
        for obs in test_obs:
            print(f"\n🧪 Testing: {obs['name']}")
            print(f"   Temperature: {obs['air_temperature']}°C")
            print(f"   Energy: {obs['HVAC_electricity_demand_rate']}W")
            print(f"   Month: {obs['month']}, Day: {obs['day_of_month']}, Hour: {obs['hour']}")
            
            result = test_reward_calculation(reward_fn, obs)
            
            if result['success']:
                print(f"   ✅ Total Reward: {result['total_reward']:.4f}")
                print(f"   📊 Energy Term: {result['energy_term']:.4f}")
                print(f"   🌡️ Temperature Term: {result['temperature_term']:.4f}")
                
                config_results[obs['name']] = result
            else:
                print(f"   ❌ Error: {result['error']}")
        
        results[config_name] = config_results
    
    # Print comparison summary
    print_summary_comparison(results, configs)

def print_summary_comparison(results: Dict, configs: Dict):
    """Print a summary comparison of all configurations."""
    
    print(f"\n{'='*80}")
    print("📊 SUMMARY COMPARISON OF ALL CONFIGURATIONS")
    print(f"{'='*80}")
    
    # Get test case names
    test_cases = list(next(iter(results.values())).keys())
    
    # Print header
    config_names = list(configs.keys())
    header = f"{'Test Case':<25}"
    for config_name in config_names:
        header += f"{config_name[:15]:<15}"
    print(header)
    print("-" * (25 + 15 * len(config_names)))
    
    # Print results for each test case
    for test_case in test_cases:
        row = f"{test_case:<25}"
        for config_name in config_names:
            if test_case in results[config_name]:
                reward = results[config_name][test_case]['total_reward']
                row += f"{reward:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    # Print configuration characteristics
    print(f"\n{'='*80}")
    print("⚙️ CONFIGURATION CHARACTERISTICS")
    print(f"{'='*80}")
    
    for config_name, config in configs.items():
        comfort_weight = 1 - config['energy_weight']
        energy_penalty = config['energy_weight'] * config['lambda_energy']
        comfort_penalty = comfort_weight * config['lambda_temperature']
        
        print(f"\n{config['name']}:")
        print(f"  Energy Focus: {config['energy_weight']*100:.0f}%")
        print(f"  Comfort Focus: {comfort_weight*100:.0f}%")
        print(f"  Energy Penalty Coeff: {energy_penalty:.2e}")
        print(f"  Comfort Penalty Coeff: {comfort_penalty:.2f}")
        print(f"  Comfort/Energy Ratio: {comfort_penalty/energy_penalty:.2e}")

# =============================================================================
# STEP 5: INTERACTIVE TESTING
# =============================================================================

def interactive_reward_tester():
    """Interactive tool to test custom reward configurations."""
    
    print("\n🎛️ INTERACTIVE REWARD CONFIGURATION TESTER")
    print("=" * 60)
    
    configs = create_reward_configurations()
    test_obs = create_test_observations()
    
    print("\nAvailable Configurations:")
    for i, (key, config) in enumerate(configs.items(), 1):
        print(f"{i}. {key}: {config['name']}")
    
    print("\nAvailable Test Cases:")
    for i, obs in enumerate(test_obs, 1):
        print(f"{i}. {obs['name']}: {obs['air_temperature']}°C, {obs['HVAC_electricity_demand_rate']}W")
    
    try:
        config_choice = input("\nSelect configuration (1-5): ")
        test_choice = input("Select test case (1-6): ")
        
        config_keys = list(configs.keys())
        config_key = config_keys[int(config_choice) - 1]
        test_obs_selected = test_obs[int(test_choice) - 1]
        
        print(f"\nTesting: {configs[config_key]['name']} with {test_obs_selected['name']}")
        
        # Create reward function
        reward_fn = create_reward_function(configs[config_key])
        
        # Test calculation
        result = test_reward_calculation(reward_fn, test_obs_selected)
        
        if result['success']:
            print(f"\n✅ Results:")
            print(f"   Total Reward: {result['total_reward']:.4f}")
            print(f"   Energy Term: {result['energy_term']:.4f}")
            print(f"   Temperature Term: {result['temperature_term']:.4f}")
        else:
            print(f"\n❌ Error: {result['error']}")
            
    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the reward testing demonstration."""
    
    print("🚀 Starting Simple Reward Function Testing")
    print("This demo shows how to customize reward weights in Sinergym")
    print("=" * 80)
    
    try:
        # Run comprehensive comparison
        compare_reward_configurations()
        
        # Run interactive tester
        interactive_reward_tester()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey Takeaways:")
        print("• energy_weight controls the balance between energy and comfort")
        print("• lambda_energy scales the energy consumption penalty")
        print("• lambda_temperature scales the comfort violation penalty")
        print("• Comfort ranges can be customized for winter/summer")
        print("• All changes are verified through reward calculation testing")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()