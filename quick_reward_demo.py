#!/usr/bin/env python3
"""
Quick Reward Weight Demo - Sinergym 5Zone Environment
=====================================================

This script demonstrates how to customize reward weights in the 5Zone environment:
- energy_weight: Controls balance between energy and comfort (0.0 to 1.0)
- lambda_energy: Scales energy consumption penalty
- lambda_temperature: Scales comfort violation penalty

Run this to see how different configurations affect reward calculations.
"""

import numpy as np
from sinergym.utils.rewards import LinearReward

def create_reward_function(energy_weight, lambda_energy, lambda_temperature):
    """Create a reward function with custom weights."""
    
    return LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(20.0, 23.5),
        range_comfort_summer=(23.0, 26.0),
        energy_weight=energy_weight,
        lambda_energy=lambda_energy,
        lambda_temperature=lambda_temperature
    )

def test_reward_calculation(reward_fn, temp, energy, month, day, hour):
    """Test reward calculation for given conditions."""
    
    obs = {
        'air_temperature': temp,
        'HVAC_electricity_demand_rate': energy,
        'month': month,
        'day_of_month': day,
        'hour': hour
    }
    
    try:
        reward, terms = reward_fn(obs)
        return reward, terms
    except Exception as e:
        return None, str(e)

def main():
    """Main demonstration function."""
    
    print("🚀 Quick Reward Weight Demo")
    print("=" * 50)
    
    # Test case: Hot summer day with high energy usage
    test_temp = 28.0  # Too hot (above summer comfort range)
    test_energy = 8000.0  # High energy usage
    test_month = 7  # July (summer)
    test_day = 15
    test_hour = 14
    
    print(f"\n🧪 Test Conditions:")
    print(f"   Temperature: {test_temp}°C (Summer comfort range: 23.0-26.0°C)")
    print(f"   Energy Usage: {test_energy}W")
    print(f"   Time: Month {test_month}, Day {test_day}, Hour {test_hour}")
    
    # Test different configurations
    configs = [
        {
            'name': 'Default (Balanced)',
            'energy_weight': 0.5,
            'lambda_energy': 1e-4,
            'lambda_temperature': 1.0
        },
        {
            'name': 'Energy-Focused',
            'energy_weight': 0.8,
            'lambda_energy': 2e-4,
            'lambda_temperature': 0.5
        },
        {
            'name': 'Comfort-Focused',
            'energy_weight': 0.2,
            'lambda_energy': 5e-5,
            'lambda_temperature': 2.0
        },
        {
            'name': 'Extreme Energy Saving',
            'energy_weight': 0.9,
            'lambda_energy': 5e-4,
            'lambda_temperature': 0.1
        },
        {
            'name': 'Extreme Comfort Priority',
            'energy_weight': 0.1,
            'lambda_energy': 1e-5,
            'lambda_temperature': 5.0
        }
    ]
    
    print(f"\n📊 Reward Calculation Results:")
    print("-" * 80)
    print(f"{'Configuration':<25} {'Energy Weight':<12} {'Lambda Energy':<15} {'Lambda Temp':<12} {'Reward':<10}")
    print("-" * 80)
    
    for config in configs:
        # Create reward function
        reward_fn = create_reward_function(
            config['energy_weight'],
            config['lambda_energy'],
            config['lambda_temperature']
        )
        
        # Calculate reward
        reward, terms = test_reward_calculation(
            reward_fn, test_temp, test_energy, test_month, test_day, test_hour
        )
        
        if reward is not None:
            print(f"{config['name']:<25} {config['energy_weight']:<12.2f} {config['lambda_energy']:<15.2e} {config['lambda_temperature']:<12.2f} {reward:<10.4f}")
        else:
            print(f"{config['name']:<25} {'ERROR':<12} {'ERROR':<15} {'ERROR':<12} {'ERROR':<10}")
    
    print(f"\n💡 Key Insights:")
    print("• Higher energy_weight = More penalty for energy usage")
    print("• Higher lambda_energy = Stronger energy penalty scaling")
    print("• Higher lambda_temperature = Stronger comfort violation penalty")
    print("• Lower energy_weight = More penalty for comfort violations")
    
    print(f"\n🎯 Configuration Effects:")
    print("• Energy-Focused: Prioritizes energy savings over comfort")
    print("• Comfort-Focused: Prioritizes comfort over energy savings")
    print("• Balanced: Equal consideration of energy and comfort")
    
    # Show detailed breakdown for one configuration
    print(f"\n🔍 Detailed Breakdown (Default Configuration):")
    default_reward_fn = create_reward_function(0.5, 1e-4, 1.0)
    reward, terms = test_reward_calculation(
        default_reward_fn, test_temp, test_energy, test_month, test_day, test_hour
    )
    
    if reward is not None and isinstance(terms, dict):
        print(f"   Total Reward: {reward:.4f}")
        print(f"   Energy Term: {terms.get('energy_term', 'N/A')}")
        print(f"   Temperature Term: {terms.get('temperature_term', 'N/A')}")
        print(f"   Formula: R = -0.5 × 1e-4 × energy - 0.5 × 1.0 × temp_violation")

if __name__ == "__main__":
    main()