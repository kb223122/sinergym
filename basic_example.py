#!/usr/bin/env python3
"""
Basic Sinergym Example
This script demonstrates the fundamental usage of Sinergym for building energy simulation.
"""

import gymnasium as gym
import numpy as np
import sinergym

def main():
    print("=== Sinergym Basic Example ===\n")
    
    # 1. Check available environments
    print("1. Available Environments:")
    available_envs = [env_id for env_id in gym.envs.registration.registry.keys() 
                     if env_id.startswith('Eplus')]
    for env_id in available_envs[:5]:  # Show first 5
        print(f"   - {env_id}")
    print(f"   ... and {len(available_envs)-5} more environments\n")
    
    # 2. Create a simple environment
    print("2. Creating Demo Environment:")
    env = gym.make('Eplus-demo-v1')
    print(f"   Environment: {env.name}")
    print(f"   Episode length: {env.episode_length} seconds")
    print(f"   Timesteps per episode: {env.timestep_per_episode}")
    print(f"   Step size: {env.step_size} seconds\n")
    
    # 3. Examine environment spaces
    print("3. Environment Spaces:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation variables: {env.observation_variables}")
    print(f"   Action variables: {env.action_variables}\n")
    
    # 4. Run a simple episode
    print("4. Running Episode with Random Actions:")
    obs, info = env.reset()
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial observation: {obs}")
    
    rewards = []
    energy_consumption = []
    comfort_violations = []
    
    terminated = truncated = False
    step_count = 0
    
    while not (terminated or truncated):
        # Take random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data
        rewards.append(reward)
        energy_consumption.append(info.get('total_power_demand', 0))
        comfort_violations.append(info.get('total_temperature_violation', 0))
        
        step_count += 1
        
        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"   Step {step_count}: Reward={reward:.2f}, "
                  f"Energy={info.get('total_power_demand', 0):.2f} kW, "
                  f"Comfort_violation={info.get('total_temperature_violation', 0):.2f}")
    
    # 5. Episode results
    print(f"\n5. Episode Results:")
    print(f"   Total steps: {step_count}")
    print(f"   Total reward: {sum(rewards):.2f}")
    print(f"   Average reward: {np.mean(rewards):.2f}")
    print(f"   Total energy consumption: {sum(energy_consumption):.2f} kWh")
    print(f"   Average energy consumption: {np.mean(energy_consumption):.2f} kW")
    print(f"   Total comfort violations: {sum(comfort_violations):.2f}")
    print(f"   Average comfort violations: {np.mean(comfort_violations):.2f}")
    
    # 6. Close environment
    env.close()
    print(f"\n6. Environment closed successfully!")
    
    # 7. Show output files
    print(f"\n7. Generated Output Files:")
    print(f"   Episode directory: {env.episode_path}")
    print(f"   Building file: {env.building_path}")
    print(f"   Weather file: {env.weather_path}")

if __name__ == "__main__":
    main()