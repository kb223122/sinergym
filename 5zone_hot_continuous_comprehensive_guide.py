#!/usr/bin/env python3
"""
5Zone Hot Continuous Environment - Comprehensive Guide
=======================================================

This script demonstrates everything you need to know about the Sinergym 5Zone hot continuous environment:
- Environment creation and configuration
- Observation space details (variables, units, ranges)
- Action space details (setpoints, units, ranges)
- Reward function customization and parameters
- Episode structure and workflow
- Complete interaction examples
- Performance analysis and monitoring

Author: AI Assistant
Date: 2024
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sinergym
from sinergym.utils.rewards import *
from typing import Dict, List, Tuple, Any
import json
import time
from datetime import datetime, timedelta

def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_subsection_header(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*60}")
    print(f"{title}")
    print(f"{'-'*60}")

class FiveZoneAnalyzer:
    """Comprehensive analyzer for 5Zone Hot Continuous Environment."""
    
    def __init__(self):
        self.env_id = 'Eplus-5zone-hot-continuous-v1'
        self.env = None
        self.episode_data = []
        self.rewards_history = []
        self.actions_history = []
        self.observations_history = []
        
    def create_environment(self, custom_reward_kwargs: Dict = None):
        """Create and configure the 5Zone environment."""
        print_section_header("1. ENVIRONMENT CREATION & CONFIGURATION")
        
        print("Available 5Zone environments:")
        available_5zone = [env_id for env_id in sinergym.__ids__ if '5zone' in env_id.lower() and 'hot' in env_id.lower()]
        for i, env_id in enumerate(available_5zone, 1):
            print(f"  {i}. {env_id}")
        
        print(f"\nCreating environment: {self.env_id}")
        
        if custom_reward_kwargs:
            # Create with custom reward parameters
            self.env = gym.make(self.env_id)
            # Note: In a real implementation, you would modify the environment configuration
            # For demonstration, we'll show how to customize rewards in the analysis
        else:
            self.env = gym.make(self.env_id)
        
        print(f"✓ Environment created successfully!")
        print(f"  Name: {self.env.name}")
        print(f"  Building file: {self.env.building_file}")
        print(f"  Weather files: {self.env.weather_files}")
        
        return self.env
    
    def analyze_observation_space(self):
        """Analyze and display detailed observation space information."""
        print_section_header("2. OBSERVATION SPACE ANALYSIS")
        
        print_subsection_header("2.1 Observation Space Overview")
        obs_space = self.env.observation_space
        print(f"  Type: {type(obs_space)}")
        print(f"  Shape: {obs_space.shape}")
        print(f"  Dtype: {obs_space.dtype}")
        print(f"  Low bounds: {obs_space.low}")
        print(f"  High bounds: {obs_space.high}")
        
        print_subsection_header("2.2 Observation Variables Details")
        obs_vars = self.env.observation_variables
        print(f"  Total variables: {len(obs_vars)}")
        
        # Group variables by category
        time_vars = self.env.time_variables
        output_vars = list(self.env.variables.keys())
        meter_vars = list(self.env.meters.keys())
        
        print(f"\n  Time Variables ({len(time_vars)}):")
        for i, var in enumerate(time_vars):
            print(f"    {i+1:2d}. {var:30s} - Current time component")
        
        print(f"\n  Output Variables ({len(output_vars)}):")
        variable_details = {
            'outdoor_temperature': ('°C', 'Site outdoor air temperature'),
            'outdoor_humidity': ('%', 'Site outdoor relative humidity'),
            'wind_speed': ('m/s', 'Site wind speed'),
            'wind_direction': ('degrees', 'Site wind direction'),
            'diffuse_solar_radiation': ('W/m²', 'Site diffuse solar radiation'),
            'direct_solar_radiation': ('W/m²', 'Site direct solar radiation'),
            'htg_setpoint': ('°C', 'Zone heating setpoint temperature'),
            'clg_setpoint': ('°C', 'Zone cooling setpoint temperature'),
            'air_temperature': ('°C', 'Zone air temperature'),
            'air_humidity': ('%', 'Zone air relative humidity'),
            'people_occupant': ('people', 'Zone occupant count'),
            'co2_emission': ('kg', 'Total CO2 emissions'),
            'HVAC_electricity_demand_rate': ('W', 'HVAC electricity demand')
        }
        
        for i, var in enumerate(output_vars):
            unit, desc = variable_details.get(var, ('', 'No description available'))
            print(f"    {i+1:2d}. {var:30s} [{unit:6s}] - {desc}")
        
        print(f"\n  Meter Variables ({len(meter_vars)}):")
        for i, var in enumerate(meter_vars):
            print(f"    {i+1:2d}. {var:30s} [J] - Energy consumption meter")
        
        # Get sample observation
        print_subsection_header("2.3 Sample Observation")
        obs, info = self.env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Sample values:")
        for i, (var, val) in enumerate(zip(obs_vars, obs)):
            unit = variable_details.get(var, ('', ''))[0]
            print(f"    {i+1:2d}. {var:30s}: {val:10.2f} [{unit}]")
    
    def analyze_action_space(self):
        """Analyze and display detailed action space information."""
        print_section_header("3. ACTION SPACE ANALYSIS")
        
        action_space = self.env.action_space
        action_vars = self.env.action_variables
        
        print_subsection_header("3.1 Action Space Overview")
        print(f"  Type: {type(action_space)}")
        print(f"  Shape: {action_space.shape}")
        print(f"  Dtype: {action_space.dtype}")
        print(f"  Low bounds: {action_space.low}")
        print(f"  High bounds: {action_space.high}")
        
        print_subsection_header("3.2 Action Variables Details")
        print(f"  Total action variables: {len(action_vars)}")
        
        actuator_details = {
            'Heating_Setpoint_RL': ('°C', 'Heating setpoint temperature', '12.0 - 23.25'),
            'Cooling_Setpoint_RL': ('°C', 'Cooling setpoint temperature', '23.25 - 30.0')
        }
        
        for i, var in enumerate(action_vars):
            unit, desc, range_str = actuator_details.get(var, ('', 'No description', 'Unknown'))
            low_val = action_space.low[i]
            high_val = action_space.high[i]
            print(f"    {i+1}. {var:25s} [{unit:3s}] - {desc}")
            print(f"       Range: {low_val:.2f} to {high_val:.2f} {unit}")
        
        print_subsection_header("3.3 Action Space Examples")
        print("  Sample random actions:")
        for i in range(5):
            action = action_space.sample()
            print(f"    Action {i+1}: Heating={action[0]:.2f}°C, Cooling={action[1]:.2f}°C")
        
        print("\n  Typical control strategies:")
        print("    Winter comfort:  Heating=20.0°C, Cooling=26.0°C")
        print("    Summer comfort:  Heating=18.0°C, Cooling=24.0°C")
        print("    Energy saving:   Heating=16.0°C, Cooling=28.0°C")
    
    def analyze_reward_function(self):
        """Analyze the reward function and show customization options."""
        print_section_header("4. REWARD FUNCTION ANALYSIS")
        
        print_subsection_header("4.1 Default Reward Function")
        reward_fn = self.env.reward_fn
        print(f"  Reward class: {type(reward_fn).__name__}")
        
        if isinstance(reward_fn, LinearReward):
            print(f"  Temperature variables: {reward_fn.temp_names}")
            print(f"  Energy variables: {reward_fn.energy_names}")
            print(f"  Winter comfort range: {reward_fn.range_comfort_winter}")
            print(f"  Summer comfort range: {reward_fn.range_comfort_summer}")
            print(f"  Summer period: {reward_fn.summer_start} to {reward_fn.summer_final}")
            print(f"  Energy weight: {reward_fn.W_energy}")
            print(f"  Lambda energy: {reward_fn.lambda_energy}")
            print(f"  Lambda temperature: {reward_fn.lambda_temp}")
        
        print_subsection_header("4.2 Reward Calculation Formula")
        print("  Linear Reward Formula:")
        print("    R = -W × λE × power - (1-W) × λT × temperature_violation")
        print("  Where:")
        print("    - W: energy_weight (0-1)")
        print("    - λE: lambda_energy (scaling factor)")
        print("    - λT: lambda_temperature (scaling factor)")
        print("    - power: total energy consumption")
        print("    - temperature_violation: deviation from comfort range")
        
        print_subsection_header("4.3 Available Reward Classes")
        reward_classes = [
            ('LinearReward', 'Standard linear combination of energy and comfort'),
            ('EnergyCostLinearReward', 'Includes energy cost in addition to consumption'),
            ('ExpReward', 'Exponential penalty for temperature violations'),
            ('HourlyLinearReward', 'Time-dependent energy/comfort weighting'),
            ('NormalizedLinearReward', 'Normalized reward with maximum penalties'),
            ('MultiZoneReward', 'Reward for multi-zone environments')
        ]
        
        for name, desc in reward_classes:
            print(f"    • {name:25s}: {desc}")
        
        print_subsection_header("4.4 Custom Reward Configuration Examples")
        self.show_custom_reward_examples()
    
    def show_custom_reward_examples(self):
        """Show examples of custom reward configurations."""
        print("\n  Example 1: Energy-focused reward")
        energy_focused = {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (18.0, 25.0),
            'range_comfort_summer': (22.0, 28.0),
            'energy_weight': 0.8,  # High energy weight
            'lambda_energy': 1e-4,
            'lambda_temperature': 0.5
        }
        print("    ", json.dumps(energy_focused, indent=6))
        
        print("\n  Example 2: Comfort-focused reward")
        comfort_focused = {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (20.0, 23.0),  # Tighter range
            'range_comfort_summer': (23.0, 26.0),  # Tighter range
            'energy_weight': 0.2,  # Low energy weight
            'lambda_energy': 1e-4,
            'lambda_temperature': 2.0  # Higher temperature penalty
        }
        print("    ", json.dumps(comfort_focused, indent=6))
        
        print("\n  Example 3: Hourly reward (time-dependent)")
        hourly_config = {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0),
            'default_energy_weight': 0.5,
            'range_comfort_hours': (8, 18),  # Comfort important 8AM-6PM
            'lambda_energy': 1e-4,
            'lambda_temperature': 1.0
        }
        print("    ", json.dumps(hourly_config, indent=6))
    
    def analyze_episode_structure(self):
        """Analyze episode structure and temporal characteristics."""
        print_section_header("5. EPISODE STRUCTURE ANALYSIS")
        
        print_subsection_header("5.1 Temporal Configuration")
        print(f"  Episode length: {self.env.episode_length} seconds")
        print(f"  Timesteps per episode: {self.env.timestep_per_episode}")
        print(f"  Step size: {self.env.step_size} seconds")
        print(f"  Steps per hour: {3600 / self.env.step_size}")
        print(f"  Steps per day: {24 * 3600 / self.env.step_size}")
        
        # Calculate episode duration
        episode_hours = self.env.episode_length / 3600
        episode_days = episode_hours / 24
        print(f"  Episode duration: {episode_hours:.1f} hours ({episode_days:.1f} days)")
        
        print_subsection_header("5.2 Simulation Period")
        runperiod = self.env.runperiod
        print(f"  Start: Month {runperiod['start_month']}, Day {runperiod['start_day']}")
        print(f"  End: Month {runperiod['end_month']}, Day {runperiod['end_day']}")
        print(f"  Year: {runperiod['start_year']}")
        
        print_subsection_header("5.3 Weather Information")
        weather_files = self.env.weather_files
        print(f"  Weather file(s): {len(weather_files)}")
        for i, weather in enumerate(weather_files):
            print(f"    {i+1}. {weather}")
        
        print_subsection_header("5.4 Building Information")
        print(f"  Building file: {self.env.building_file}")
        print(f"  Zone names: {self.env.zone_names}")
    
    def demonstrate_workflow(self, num_episodes: int = 2, steps_per_episode: int = 100):
        """Demonstrate complete workflow with multiple episodes."""
        print_section_header("6. COMPLETE WORKFLOW DEMONSTRATION")
        
        print_subsection_header("6.1 Multi-Episode Simulation")
        
        total_rewards = []
        episode_stats = []
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            
            # Reset environment
            obs, info = self.env.reset()
            episode_reward = 0
            episode_energy = 0
            episode_comfort_violations = 0
            episode_actions = []
            episode_observations = []
            
            print(f"Initial observation shape: {obs.shape}")
            print(f"Initial info keys: {list(info.keys())}")
            
            # Run episode
            for step in range(steps_per_episode):
                # Choose action (random for demonstration)
                action = self.env.action_space.sample()
                episode_actions.append(action.copy())
                episode_observations.append(obs.copy())
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Collect metrics
                episode_reward += reward
                episode_energy += info.get('total_power_demand', 0)
                episode_comfort_violations += info.get('total_temperature_violation', 0)
                
                # Print progress
                if step % 20 == 0:
                    print(f"  Step {step:3d}: Reward={reward:6.2f}, "
                          f"Energy={info.get('total_power_demand', 0):6.1f}W, "
                          f"Comfort={info.get('total_temperature_violation', 0):4.2f}°C")
                
                if terminated or truncated:
                    print(f"  Episode ended at step {step}")
                    break
            
            # Episode statistics
            stats = {
                'episode': episode + 1,
                'total_reward': episode_reward,
                'avg_reward': episode_reward / (step + 1),
                'total_energy': episode_energy,
                'avg_energy': episode_energy / (step + 1),
                'total_comfort_violations': episode_comfort_violations,
                'avg_comfort_violations': episode_comfort_violations / (step + 1),
                'steps': step + 1
            }
            episode_stats.append(stats)
            total_rewards.append(episode_reward)
            
            print(f"Episode {episode + 1} Results:")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Average reward: {episode_reward / (step + 1):.2f}")
            print(f"  Total energy: {episode_energy:.2f} Wh")
            print(f"  Average comfort violations: {episode_comfort_violations / (step + 1):.2f}°C")
        
        print_subsection_header("6.2 Performance Summary")
        print(f"Total episodes: {num_episodes}")
        print(f"Average episode reward: {np.mean(total_rewards):.2f}")
        print(f"Best episode reward: {max(total_rewards):.2f}")
        print(f"Worst episode reward: {min(total_rewards):.2f}")
        
        return episode_stats
    
    def demonstrate_reward_comparison(self):
        """Demonstrate different reward functions with the same actions."""
        print_section_header("7. REWARD FUNCTION COMPARISON")
        
        # Sample observation for testing
        obs, _ = self.env.reset()
        obs_dict = dict(zip(self.env.observation_variables, obs))
        
        print_subsection_header("7.1 Sample Observation")
        for var, val in obs_dict.items():
            print(f"  {var}: {val:.2f}")
        
        print_subsection_header("7.2 Reward Comparison")
        
        # Test different reward functions
        reward_configs = [
            {
                'name': 'Standard Linear',
                'class': LinearReward,
                'kwargs': {
                    'temperature_variables': ['air_temperature'],
                    'energy_variables': ['HVAC_electricity_demand_rate'],
                    'range_comfort_winter': (20.0, 23.5),
                    'range_comfort_summer': (23.0, 26.0),
                    'energy_weight': 0.5
                }
            },
            {
                'name': 'Energy Focused',
                'class': LinearReward,
                'kwargs': {
                    'temperature_variables': ['air_temperature'],
                    'energy_variables': ['HVAC_electricity_demand_rate'],
                    'range_comfort_winter': (20.0, 23.5),
                    'range_comfort_summer': (23.0, 26.0),
                    'energy_weight': 0.8
                }
            },
            {
                'name': 'Comfort Focused',
                'class': LinearReward,
                'kwargs': {
                    'temperature_variables': ['air_temperature'],
                    'energy_variables': ['HVAC_electricity_demand_rate'],
                    'range_comfort_winter': (20.0, 23.5),
                    'range_comfort_summer': (23.0, 26.0),
                    'energy_weight': 0.2
                }
            },
            {
                'name': 'Exponential',
                'class': ExpReward,
                'kwargs': {
                    'temperature_variables': ['air_temperature'],
                    'energy_variables': ['HVAC_electricity_demand_rate'],
                    'range_comfort_winter': (20.0, 23.5),
                    'range_comfort_summer': (23.0, 26.0),
                    'energy_weight': 0.5
                }
            }
        ]
        
        for config in reward_configs:
            try:
                reward_fn = config['class'](**config['kwargs'])
                reward, reward_terms = reward_fn(obs_dict)
                print(f"\n  {config['name']}:")
                print(f"    Total reward: {reward:.4f}")
                print(f"    Energy term: {reward_terms.get('energy_term', 'N/A'):.4f}")
                print(f"    Comfort term: {reward_terms.get('comfort_term', 'N/A'):.4f}")
                print(f"    Power demand: {reward_terms.get('total_power_demand', 'N/A'):.2f} W")
                print(f"    Temp violations: {reward_terms.get('total_temperature_violation', 'N/A'):.2f} °C")
            except Exception as e:
                print(f"  {config['name']}: Error - {e}")
    
    def show_advanced_features(self):
        """Show advanced features and customization options."""
        print_section_header("8. ADVANCED FEATURES & CUSTOMIZATION")
        
        print_subsection_header("8.1 Weather Variability")
        print("  Add stochastic weather variations using Ornstein-Uhlenbeck process:")
        weather_variability_example = {
            'Dry Bulb Temperature': (1.0, 0.0, 24.0),  # sigma, mu, tau
            'Relative Humidity': (5.0, 0.0, 12.0),
            'Wind Speed': (0.5, 0.0, 6.0)
        }
        print("    Example configuration:")
        print("   ", json.dumps(weather_variability_example, indent=6))
        
        print_subsection_header("8.2 Context Variables")
        context_vars = self.env.context_variables
        print(f"  Context variables: {context_vars}")
        print("  Use context variables for non-agent controlled actuators")
        print("  Example: Fixed schedules, external controls, occupancy patterns")
        
        print_subsection_header("8.3 Environment Wrappers")
        print("  Available wrappers for enhanced functionality:")
        wrapper_examples = [
            ('LoggerWrapper', 'Log all interactions to CSV files'),
            ('MultiObjectiveReward', 'Handle multiple objective functions'),
            ('NormalizeObservation', 'Normalize observations to [0,1] range'),
            ('DiscretizeEnv', 'Convert continuous actions to discrete'),
            ('PreviousObservationWrapper', 'Include previous observations'),
            ('DatetimeWrapper', 'Add datetime information to observations')
        ]
        for name, desc in wrapper_examples:
            print(f"    • {name:25s}: {desc}")
        
        print_subsection_header("8.4 Custom Building Models")
        print("  Create custom building models by:")
        print("    1. Modifying .epJSON building files")
        print("    2. Adjusting HVAC systems and controls")
        print("    3. Changing building geometry and materials")
        print("    4. Adding new zones and equipment")
        
        print_subsection_header("8.5 Multi-Zone Extensions")
        print("  Extend to multi-zone control:")
        print("    • Multiple temperature setpoints per zone")
        print("    • Zone-specific comfort ranges")
        print("    • Independent HVAC systems")
        print("    • Inter-zone heat transfer considerations")
    
    def generate_performance_report(self, episode_stats: List[Dict]):
        """Generate a comprehensive performance report."""
        print_section_header("9. PERFORMANCE ANALYSIS REPORT")
        
        if not episode_stats:
            print("No episode data available for analysis.")
            return
        
        df = pd.DataFrame(episode_stats)
        
        print_subsection_header("9.1 Statistical Summary")
        print(f"{'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 65)
        
        metrics = ['total_reward', 'total_energy', 'total_comfort_violations']
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                min_val = df[metric].min()
                max_val = df[metric].max()
                print(f"{metric:<25} {mean_val:<10.2f} {std_val:<10.2f} {min_val:<10.2f} {max_val:<10.2f}")
        
        print_subsection_header("9.2 Performance Insights")
        
        # Energy efficiency analysis
        avg_energy = df['total_energy'].mean()
        print(f"  Average energy consumption: {avg_energy:.2f} Wh per episode")
        
        # Comfort analysis
        avg_comfort = df['total_comfort_violations'].mean()
        print(f"  Average comfort violations: {avg_comfort:.2f} °C per episode")
        
        # Reward analysis
        avg_reward = df['total_reward'].mean()
        print(f"  Average episode reward: {avg_reward:.2f}")
        
        if avg_energy > 0:
            comfort_efficiency = avg_comfort / avg_energy * 1000
            print(f"  Comfort efficiency: {comfort_efficiency:.4f} °C violations per kWh")
        
        print_subsection_header("9.3 Optimization Recommendations")
        
        if avg_energy > 1000:  # Arbitrary threshold
            print("  • High energy consumption detected")
            print("    - Consider increasing energy_weight in reward function")
            print("    - Implement more aggressive energy-saving strategies")
            print("    - Review HVAC system efficiency")
        
        if avg_comfort > 10:  # Arbitrary threshold
            print("  • High comfort violations detected")
            print("    - Consider decreasing energy_weight in reward function")
            print("    - Tighten temperature setpoint control")
            print("    - Review comfort range settings")
        
        if avg_reward < -100:  # Arbitrary threshold
            print("  • Low reward scores detected")
            print("    - Review reward function parameters")
            print("    - Consider different reward formulation")
            print("    - Analyze trade-off between energy and comfort")
    
    def cleanup(self):
        """Clean up environment resources."""
        if self.env:
            self.env.close()
            print("\n✓ Environment closed successfully!")

def main():
    """Main function demonstrating comprehensive 5Zone environment analysis."""
    
    print_section_header("SINERGYM 5ZONE HOT CONTINUOUS ENVIRONMENT")
    print("COMPREHENSIVE ANALYSIS AND DEMONSTRATION")
    print("\nThis script provides complete information about:")
    print("• Environment configuration and spaces")
    print("• Observation variables with units and descriptions")  
    print("• Action space and control variables")
    print("• Reward function analysis and customization")
    print("• Episode structure and temporal characteristics")
    print("• Complete workflow demonstration")
    print("• Performance analysis and optimization")
    
    # Create analyzer instance
    analyzer = FiveZoneAnalyzer()
    
    try:
        # 1. Create environment
        env = analyzer.create_environment()
        
        # 2. Analyze observation space
        analyzer.analyze_observation_space()
        
        # 3. Analyze action space
        analyzer.analyze_action_space()
        
        # 4. Analyze reward function
        analyzer.analyze_reward_function()
        
        # 5. Analyze episode structure
        analyzer.analyze_episode_structure()
        
        # 6. Demonstrate workflow
        episode_stats = analyzer.demonstrate_workflow(num_episodes=2, steps_per_episode=50)
        
        # 7. Demonstrate reward comparison
        analyzer.demonstrate_reward_comparison()
        
        # 8. Show advanced features
        analyzer.show_advanced_features()
        
        # 9. Generate performance report
        analyzer.generate_performance_report(episode_stats)
        
        print_section_header("SUMMARY")
        print("✓ Comprehensive analysis completed!")
        print("\nKey Takeaways:")
        print("• 5Zone environment simulates a 5-zone building with HVAC control")
        print("• Observation space includes weather, zone conditions, and energy data")
        print("• Action space controls heating and cooling setpoints (2D continuous)")
        print("• Default LinearReward balances energy consumption and thermal comfort")
        print("• Episodes can run for full simulation periods (days/weeks/months)")
        print("• Highly customizable through reward functions and configuration")
        print("• Suitable for multi-objective HVAC optimization research")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        analyzer.cleanup()

if __name__ == "__main__":
    main()