#!/usr/bin/env python3
"""
Rule-Based Controller Example
This script demonstrates how to implement rule-based control strategies for building energy optimization.
"""

import gymnasium as gym
import numpy as np
import sinergym

class RuleBasedController:
    """Simple rule-based controller for building energy optimization."""
    
    def __init__(self, strategy='balanced'):
        """
        Initialize the controller.
        
        Args:
            strategy (str): Control strategy ('energy_saving', 'comfort_focused', 'balanced')
        """
        self.strategy = strategy
        
    def get_action(self, obs):
        """
        Generate control action based on current observations.
        
        Args:
            obs (np.ndarray): Current observation from environment
            
        Returns:
            np.ndarray: Control action [heating_setpoint, cooling_setpoint]
        """
        # Extract relevant observations (indices may vary by environment)
        outdoor_temp = obs[0]  # Site Outdoor Air DryBulb Temperature
        indoor_temp = obs[8]   # Zone Air Temperature (assuming 9th observation)
        
        if self.strategy == 'energy_saving':
            return self._energy_saving_strategy(outdoor_temp, indoor_temp)
        elif self.strategy == 'comfort_focused':
            return self._comfort_focused_strategy(outdoor_temp, indoor_temp)
        else:  # balanced
            return self._balanced_strategy(outdoor_temp, indoor_temp)
    
    def _energy_saving_strategy(self, outdoor_temp, indoor_temp):
        """Energy-saving strategy with wider comfort bands."""
        # Heating setpoint: lower when outdoor temp is higher
        if outdoor_temp < 10:
            heating_setpoint = 18.0
        elif outdoor_temp < 15:
            heating_setpoint = 19.0
        else:
            heating_setpoint = 20.0
        
        # Cooling setpoint: higher when outdoor temp is lower
        if outdoor_temp > 30:
            cooling_setpoint = 26.0
        elif outdoor_temp > 25:
            cooling_setpoint = 27.0
        else:
            cooling_setpoint = 28.0
        
        return np.array([heating_setpoint, cooling_setpoint], dtype=np.float32)
    
    def _comfort_focused_strategy(self, outdoor_temp, indoor_temp):
        """Comfort-focused strategy with tight temperature control."""
        # Always maintain comfortable temperature range
        heating_setpoint = 21.0
        cooling_setpoint = 25.0
        
        return np.array([heating_setpoint, cooling_setpoint], dtype=np.float32)
    
    def _balanced_strategy(self, outdoor_temp, indoor_temp):
        """Balanced strategy between energy and comfort."""
        # Adaptive setpoints based on outdoor temperature
        if outdoor_temp < 15:
            heating_setpoint = 20.0
        else:
            heating_setpoint = 19.0
        
        if outdoor_temp > 25:
            cooling_setpoint = 25.0
        else:
            cooling_setpoint = 26.0
        
        return np.array([heating_setpoint, cooling_setpoint], dtype=np.float32)

def evaluate_controller(env, controller, episode_name):
    """Evaluate a controller on the given environment."""
    print(f"\n=== Evaluating {episode_name} ===")
    
    obs, info = env.reset()
    rewards = []
    energy_consumption = []
    comfort_violations = []
    actions_taken = []
    
    terminated = truncated = False
    step_count = 0
    
    while not (terminated or truncated):
        # Get action from controller
        action = controller.get_action(obs)
        actions_taken.append(action.copy())
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data
        rewards.append(reward)
        energy_consumption.append(info.get('total_power_demand', 0))
        comfort_violations.append(info.get('total_temperature_violation', 0))
        
        step_count += 1
        
        # Print progress every 200 steps
        if step_count % 200 == 0:
            print(f"   Step {step_count}: Reward={reward:.2f}, "
                  f"Energy={info.get('total_power_demand', 0):.2f} kW, "
                  f"Action={action}")
    
    # Calculate metrics
    total_reward = sum(rewards)
    avg_reward = np.mean(rewards)
    total_energy = sum(energy_consumption)
    avg_energy = np.mean(energy_consumption)
    total_comfort_violation = sum(comfort_violations)
    avg_comfort_violation = np.mean(comfort_violations)
    
    # Print results
    print(f"Results for {episode_name}:")
    print(f"   Total steps: {step_count}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Total energy consumption: {total_energy:.2f} kWh")
    print(f"   Average energy consumption: {avg_energy:.2f} kW")
    print(f"   Total comfort violations: {total_comfort_violation:.2f}")
    print(f"   Average comfort violations: {avg_comfort_violation:.2f}")
    
    return {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'total_energy': total_energy,
        'avg_energy': avg_energy,
        'total_comfort_violation': total_comfort_violation,
        'avg_comfort_violation': avg_comfort_violation,
        'actions': actions_taken
    }

def main():
    print("=== Rule-Based Controller Example ===\n")
    
    # Create environment
    print("Creating environment...")
    env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
    print(f"Environment: {env.name}")
    print(f"Episode length: {env.episode_length} seconds")
    print(f"Timesteps per episode: {env.timestep_per_episode}")
    
    # Create controllers
    controllers = {
        'Energy Saving': RuleBasedController('energy_saving'),
        'Comfort Focused': RuleBasedController('comfort_focused'),
        'Balanced': RuleBasedController('balanced')
    }
    
    # Evaluate each controller
    results = {}
    for name, controller in controllers.items():
        results[name] = evaluate_controller(env, controller, name)
    
    # Compare results
    print("\n=== Comparison of Control Strategies ===")
    print(f"{'Strategy':<15} {'Total Reward':<12} {'Avg Energy':<10} {'Comfort Viol.':<12}")
    print("-" * 55)
    
    for name, result in results.items():
        print(f"{name:<15} {result['total_reward']:<12.2f} "
              f"{result['avg_energy']:<10.2f} {result['total_comfort_violation']:<12.2f}")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda x: results[x]['total_reward'])
    print(f"\nBest strategy (lowest total reward): {best_strategy}")
    
    # Close environment
    env.close()
    print(f"\nEnvironment closed successfully!")

if __name__ == "__main__":
    main()