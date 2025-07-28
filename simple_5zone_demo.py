#!/usr/bin/env python3
"""
Simple 5Zone Environment Customization Demo
===========================================

This script demonstrates the key concepts of customizing the 
'Eplus-5zone-hot-continuous-v1' environment without requiring
external dependencies.

Run this to understand the concepts, then use the full scripts
when you have Sinergym installed.
"""

import math
from datetime import datetime

def simulate_reward_calculation(air_temp, hvac_power, energy_weight, lambda_energy, lambda_temp, month, day_of_month=15):
    """
    Simulate the LinearReward calculation for demonstration.
    """
    # Comfort ranges
    if 6 <= month <= 9:  # Summer
        comfort_low, comfort_high = 23.0, 26.0
    else:  # Winter
        comfort_low, comfort_high = 20.0, 23.5
    
    # Calculate temperature violation
    temp_violation = 0
    if air_temp < comfort_low:
        temp_violation = comfort_low - air_temp
    elif air_temp > comfort_high:
        temp_violation = air_temp - comfort_high
    
    # Calculate reward components
    energy_penalty = lambda_energy * hvac_power
    comfort_penalty = lambda_temp * temp_violation
    
    # Combined reward
    reward = -(energy_weight * energy_penalty + (1 - energy_weight) * comfort_penalty)
    
    return reward, {
        'energy_penalty': -energy_weight * energy_penalty,
        'comfort_penalty': -(1 - energy_weight) * comfort_penalty,
        'temp_violation': temp_violation,
        'comfort_range': f"{comfort_low}-{comfort_high}°C"
    }

def test_reward_parameters():
    """
    Test different reward parameter combinations.
    """
    print("🧪 TESTING REWARD PARAMETERS")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Comfortable, Low Energy',
            'air_temp': 22.0,
            'hvac_power': 3000.0,
            'month': 4
        },
        {
            'name': 'Too Hot, High Energy',
            'air_temp': 28.0,
            'hvac_power': 8000.0,
            'month': 7
        },
        {
            'name': 'Too Cold, High Energy',
            'air_temp': 18.0,
            'hvac_power': 6000.0,
            'month': 12
        },
        {
            'name': 'Comfortable, Moderate Energy',
            'air_temp': 24.0,
            'hvac_power': 5000.0,
            'month': 5
        }
    ]
    
    # Parameter combinations
    param_combinations = [
        {
            'name': 'Default (Balanced)',
            'energy_weight': 0.5,
            'lambda_energy': 1.0e-4,
            'lambda_temp': 1.0
        },
        {
            'name': 'Energy Focused',
            'energy_weight': 0.8,
            'lambda_energy': 2.0e-4,
            'lambda_temp': 0.5
        },
        {
            'name': 'Comfort Focused',
            'energy_weight': 0.2,
            'lambda_energy': 0.5e-4,
            'lambda_temp': 2.0
        },
        {
            'name': 'High Energy Penalty',
            'energy_weight': 0.6,
            'lambda_energy': 3.0e-4,
            'lambda_temp': 1.0
        },
        {
            'name': 'High Comfort Penalty',
            'energy_weight': 0.4,
            'lambda_energy': 1.0e-4,
            'lambda_temp': 3.0
        }
    ]
    
    # Test each combination
    for params in param_combinations:
        print(f"\n📊 {params['name']}")
        print("-" * 40)
        print(f"Energy weight: {params['energy_weight']}")
        print(f"Lambda energy: {params['lambda_energy']}")
        print(f"Lambda temperature: {params['lambda_temp']}")
        print()
        
        total_reward = 0
        for scenario in scenarios:
            reward, terms = simulate_reward_calculation(
                scenario['air_temp'],
                scenario['hvac_power'],
                params['energy_weight'],
                params['lambda_energy'],
                params['lambda_temp'],
                scenario['month']
            )
            total_reward += reward
            
            print(f"  {scenario['name']}:")
            print(f"    Temp: {scenario['air_temp']}°C, Energy: {scenario['hvac_power']}W")
            print(f"    Comfort range: {terms['comfort_range']}")
            print(f"    Reward: {reward:.4f}")
            print(f"    Energy penalty: {terms['energy_penalty']:.4f}")
            print(f"    Comfort penalty: {terms['comfort_penalty']:.4f}")
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Average reward: {total_reward/len(scenarios):.4f}")

def show_environment_info():
    """
    Show information about the 5Zone environment.
    """
    print("\n🏢 5ZONE HOT ENVIRONMENT INFORMATION")
    print("=" * 50)
    
    print("📋 Environment Details:")
    print("  - Environment ID: Eplus-5zone-hot-continuous-v1")
    print("  - Building: 5-zone commercial building with VAV HVAC")
    print("  - Weather: Hot climate (Arizona)")
    print("  - Building file: 5ZoneAutoDXVAV.epJSON")
    print("  - Weather file: USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw")
    
    print("\n🎯 Action Space:")
    print("  - Type: Continuous Box")
    print("  - Action 1 (Heating Setpoint): 12.0°C to 23.25°C")
    print("  - Action 2 (Cooling Setpoint): 23.25°C to 30.0°C")
    
    print("\n📊 Observation Space:")
    print("  - Type: Continuous Box")
    print("  - Variables: 15 total")
    print("  - Time variables: month, day_of_month, hour")
    print("  - Weather variables: outdoor temperature, humidity, wind, solar radiation")
    print("  - Building variables: zone temperature, humidity, occupancy")
    print("  - Energy variables: HVAC electricity demand, CO2 emissions")
    print("  - Setpoint variables: current heating/cooling setpoints")
    
    print("\n⏰ Episode Information:")
    print("  - Default episode length: 35,040 timesteps (1 year)")
    print("  - Default timestep: 15 minutes (900 seconds)")
    print("  - Episode duration: 1 year")
    
    print("\n🎯 Reward Function (LinearReward):")
    print("  - Formula: R = -W * λ_E * Energy - (1-W) * λ_T * Temperature_Violation")
    print("  - Default energy weight (W): 0.5")
    print("  - Default lambda energy (λ_E): 1.0e-4")
    print("  - Default lambda temperature (λ_T): 1.0")
    print("  - Comfort ranges:")
    print("    * Winter (Oct-May): 20.0°C to 23.5°C")
    print("    * Summer (Jun-Sep): 23.0°C to 26.0°C")

def show_customization_examples():
    """
    Show examples of different customizations.
    """
    print("\n🔧 CUSTOMIZATION EXAMPLES")
    print("=" * 40)
    
    print("\n1️⃣ Custom Run Periods:")
    print("   Default (1 year):")
    print("     config_params={'runperiod': (1, 1, 1991, 12, 31, 1991), 'timesteps_per_hour': 1}")
    print("   ")
    print("   1 Month:")
    print("     config_params={'runperiod': (1, 1, 1991, 1, 31, 1991), 'timesteps_per_hour': 1}")
    print("   ")
    print("   1 Week:")
    print("     config_params={'runperiod': (1, 1, 1991, 1, 7, 1991), 'timesteps_per_hour': 4}")
    print("   ")
    print("   1 Day:")
    print("     config_params={'runperiod': (1, 1, 1991, 1, 1, 1991), 'timesteps_per_hour': 12}")
    
    print("\n2️⃣ Custom Reward Functions:")
    print("   Energy-focused:")
    print("     LinearReward(energy_weight=0.8, lambda_energy=2.0e-4, lambda_temperature=0.5)")
    print("   ")
    print("   Comfort-focused:")
    print("     LinearReward(energy_weight=0.2, lambda_energy=0.5e-4, lambda_temperature=2.0)")
    print("   ")
    print("   Hourly (comfort only 9 AM - 7 PM):")
    print("     HourlyLinearReward(range_comfort_hours=(9, 19))")
    
    print("\n3️⃣ Weather Variability:")
    print("   weather_variability={")
    print("     'Site Outdoor Air DryBulb Temperature': (2.0, 0.0, 24.0),  # sigma, mu, tau")
    print("     'Site Outdoor Air Relative Humidity': (5.0, 0.0, 24.0)")
    print("   }")
    
    print("\n4️⃣ Environment Creation Examples:")
    print("   # Default environment")
    print("   env = gym.make('Eplus-5zone-hot-continuous-v1')")
    print("   ")
    print("   # Custom reward and run period")
    print("   custom_reward = LinearReward(energy_weight=0.7, lambda_energy=1.5e-4)")
    print("   env = gym.make('Eplus-5zone-hot-continuous-v1',")
    print("                  reward=custom_reward,")
    print("                  config_params={'runperiod': (1, 1, 1991, 1, 7, 1991)})")

def show_training_example():
    """
    Show a training example.
    """
    print("\n🚀 TRAINING EXAMPLE")
    print("=" * 30)
    
    print("1. Create environment with custom settings:")
    print("   env = gym.make('Eplus-5zone-hot-continuous-v1',")
    print("                  env_name='CustomTraining',")
    print("                  config_params={'runperiod': (1, 1, 1991, 1, 7, 1991)})")
    print("   ")
    print("2. Add wrappers:")
    print("   env = NormalizeObservation(env)")
    print("   env = NormalizeAction(env)")
    print("   env = LoggerWrapper(env)")
    print("   env = CSVLogger(env)")
    print("   ")
    print("3. Create PPO model:")
    print("   model = PPO('MlpPolicy', env,")
    print("               learning_rate=0.0003,")
    print("               n_steps=1024,")
    print("               batch_size=64)")
    print("   ")
    print("4. Train the model:")
    print("   model.learn(total_timesteps=timesteps)")
    print("   ")
    print("5. Save and evaluate:")
    print("   model.save('trained_model')")
    print("   # Evaluate on separate environment")

def main():
    """
    Main function to run the demonstration.
    """
    print("🎯 5ZONE HOT ENVIRONMENT CUSTOMIZATION DEMO")
    print("=" * 60)
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Show environment information
    show_environment_info()
    
    # Test reward parameters
    test_reward_parameters()
    
    # Show customization examples
    show_customization_examples()
    
    # Show training example
    show_training_example()
    
    print("\n🎉 DEMONSTRATION COMPLETED!")
    print("=" * 60)
    print("📚 What you've learned:")
    print("  ✅ How reward parameters affect the reward calculation")
    print("  ✅ How to customize run periods and timesteps")
    print("  ✅ How to create different reward functions")
    print("  ✅ How to set up training environments")
    print("  ✅ How to configure PPO training")
    
    print("\n🚀 Next Steps:")
    print("  1. Install Sinergym: pip install sinergym[drl]")
    print("  2. Run the comprehensive guide: python comprehensive_5zone_customization_guide.py")
    print("  3. Test reward parameters: python test_reward_parameters.py")
    print("  4. Use custom config: python scripts/train/local_confs/train_agent_local_conf.py -conf custom_5zone_config.yaml")
    
    print("\n💡 Key Parameters to Experiment With:")
    print("  - energy_weight: 0.0-1.0 (higher = more energy focus)")
    print("  - lambda_energy: 1e-5 to 1e-3 (higher = stronger energy penalty)")
    print("  - lambda_temperature: 0.1-5.0 (higher = stronger comfort penalty)")
    print("  - runperiod: Change episode length")
    print("  - timesteps_per_hour: Change simulation resolution")

if __name__ == "__main__":
    main()