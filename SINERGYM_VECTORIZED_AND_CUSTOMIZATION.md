# Sinergym Vectorized Environments and Customization Guide

## 1. Vectorized Environment Support

### Current Status: **Limited Support**

Sinergym has **limited native support** for vectorized environments like `SubprocVecEnv`. Here's what I found:

### What's Available

#### ✅ **DummyVecEnv Support**
```python
from stable_baselines3.common.vec_env import DummyVecEnv

# Example from reinforcement_learning_example.py
def create_env(env_name, use_wrappers=True):
    env = gym.make(env_name)
    if use_wrappers:
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        env = LoggerWrapper(env)
        env = CSVLogger(env)
    return env

# Can be wrapped in DummyVecEnv
env = DummyVecEnv([lambda: create_env('Eplus-5zone-hot-continuous-v1')])
```

#### ❌ **No SubprocVecEnv Support**
- No examples found in the codebase
- Sinergym environments are **not designed for multiprocessing**
- EnergyPlus simulation is **computationally intensive** and may not benefit from parallelization

### Why Limited Vectorization?

1. **EnergyPlus Complexity**: Each environment runs a full EnergyPlus simulation
2. **Resource Intensive**: Multiple EnergyPlus processes would be very resource-heavy
3. **File System Conflicts**: Multiple processes writing to same directories
4. **Design Philosophy**: Sinergym focuses on single, detailed simulations

### Workaround for Parallel Training

```python
# Manual parallel training approach
import multiprocessing as mp
from stable_baselines3 import PPO

def train_single_env(env_id, total_timesteps=50000):
    """Train a single environment instance."""
    env = gym.make('Eplus-5zone-hot-continuous-v1')
    model = PPO('MlpPolicy', env)
    model.learn(total_timesteps=total_timesteps)
    return model

# Train multiple models in parallel
if __name__ == '__main__':
    with mp.Pool(processes=4) as pool:
        results = pool.map(train_single_env, range(4))
```

## 2. Runperiod Customization

### How Runperiod Works

Runperiod defines the **simulation time period** for each episode. It's specified as a tuple: `(start_day, start_month, start_year, end_day, end_month, end_year)`

### Default Runperiod
```python
# Default in sinergym/__init__.py
'runperiod': (1, 1, 1991, 1, 3, 1991),  # Jan 1 to Mar 1, 1991
```

### Customization Examples

#### Example 1: Full Year Simulation
```python
env_params = {
    'runperiod': [1, 1, 1991, 31, 12, 1991],  # Full year
    'timesteps_per_hour': 4
}

env = gym.make('Eplus-5zone-hot-continuous-v1', **env_params)
```

#### Example 2: Summer Season Only
```python
env_params = {
    'runperiod': [1, 6, 1991, 31, 8, 1991],  # June 1 to Aug 31
    'timesteps_per_hour': 4
}

env = gym.make('Eplus-5zone-hot-continuous-v1', **env_params)
```

#### Example 3: Winter Season Only
```python
env_params = {
    'runperiod': [1, 12, 1991, 28, 2, 1992],  # Dec 1 to Feb 28
    'timesteps_per_hour': 4
}

env = gym.make('Eplus-5zone-hot-continuous-v1', **env_params)
```

#### Example 4: Specific Month
```python
env_params = {
    'runperiod': [1, 7, 1991, 31, 7, 1991],  # July only
    'timesteps_per_hour': 4
}

env = gym.make('Eplus-5zone-hot-continuous-v1', **env_params)
```

### Runperiod Impact on Training

```python
# Calculate episode length
episode_length = env.get_wrapper_attr('timestep_per_episode')
print(f"Episode length: {episode_length} timesteps")

# For different runperiods:
# Full year (8760 hours): ~8760 timesteps
# Summer (3 months): ~2190 timesteps  
# Winter (3 months): ~2190 timesteps
# Single month: ~730 timesteps
```

## 3. Reward Weight Customization

### Available Reward Functions

Sinergym provides several reward function classes:

1. **LinearReward** (most common)
2. **ExpReward** (exponential penalties)
3. **EnergyCostLinearReward** (includes energy costs)
4. **HourlyLinearReward** (time-dependent weights)
5. **NormalizedLinearReward** (normalized penalties)
6. **MultiZoneReward** (multi-zone comfort)

### LinearReward Customization

#### Basic Structure
```python
reward_config = {
    'temperature_variables': ['air_temperature'],
    'energy_variables': ['HVAC_electricity_demand_rate'],
    'range_comfort_winter': [20.0, 23.5],
    'range_comfort_summer': [23.0, 26.0],
    'summer_start': [6, 1],
    'summer_final': [9, 30],
    'energy_weight': 0.5,
    'lambda_energy': 0.0001,
    'lambda_temperature': 1.0
}
```

#### Example 1: Energy-Focused Training
```python
env_params = {
    'runperiod': [1, 1, 1991, 31, 12, 1991],
    'reward': {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': [20.0, 23.5],
        'range_comfort_summer': [23.0, 26.0],
        'summer_start': [6, 1],
        'summer_final': [9, 30],
        'energy_weight': 0.8,        # High energy focus
        'lambda_energy': 0.0001,
        'lambda_temperature': 0.5     # Lower comfort penalty
    }
}
```

#### Example 2: Comfort-Focused Training
```python
env_params = {
    'runperiod': [1, 1, 1991, 31, 12, 1991],
    'reward': {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': [20.0, 23.5],
        'range_comfort_summer': [23.0, 26.0],
        'summer_start': [6, 1],
        'summer_final': [9, 30],
        'energy_weight': 0.2,        # Low energy focus
        'lambda_energy': 0.0001,
        'lambda_temperature': 2.0     # Higher comfort penalty
    }
}
```

#### Example 3: Balanced Training
```python
env_params = {
    'runperiod': [1, 1, 1991, 31, 12, 1991],
    'reward': {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': [20.0, 23.5],
        'range_comfort_summer': [23.0, 26.0],
        'summer_start': [6, 1],
        'summer_final': [9, 30],
        'energy_weight': 0.5,        # Balanced
        'lambda_energy': 0.0001,
        'lambda_temperature': 1.0     # Standard comfort penalty
    }
}
```

### Advanced Reward Customization

#### Custom Comfort Ranges
```python
env_params = {
    'reward': {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': [18.0, 22.0],  # Stricter winter comfort
        'range_comfort_summer': [24.0, 28.0],  # Wider summer comfort
        'summer_start': [6, 1],
        'summer_final': [9, 30],
        'energy_weight': 0.5,
        'lambda_energy': 0.0001,
        'lambda_temperature': 1.0
    }
}
```

#### Multiple Energy Variables
```python
env_params = {
    'reward': {
        'temperature_variables': ['air_temperature'],
        'energy_variables': [
            'HVAC_electricity_demand_rate',
            'HVAC_gas_demand_rate',
            'total_electricity_demand_rate'
        ],
        'range_comfort_winter': [20.0, 23.5],
        'range_comfort_summer': [23.0, 26.0],
        'summer_start': [6, 1],
        'summer_final': [9, 30],
        'energy_weight': 0.5,
        'lambda_energy': 0.0001,
        'lambda_temperature': 1.0
    }
}
```

#### Custom Summer Period
```python
env_params = {
    'reward': {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': [20.0, 23.5],
        'range_comfort_summer': [23.0, 26.0],
        'summer_start': [5, 1],      # Summer starts in May
        'summer_final': [10, 31],    # Summer ends in October
        'energy_weight': 0.5,
        'lambda_energy': 0.0001,
        'lambda_temperature': 1.0
    }
}
```

## 4. Complete Customization Example

```python
import gymnasium as gym
import sinergym
from stable_baselines3 import PPO

# Custom environment configuration
env_params = {
    # Time period: Summer months only
    'runperiod': [1, 6, 1991, 31, 8, 1991],
    
    # Simulation resolution
    'timesteps_per_hour': 4,
    
    # Custom reward function
    'reward': {
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': [20.0, 23.5],
        'range_comfort_summer': [23.0, 26.0],
        'summer_start': [6, 1],
        'summer_final': [8, 31],
        'energy_weight': 0.7,        # Focus on energy efficiency
        'lambda_energy': 0.0001,
        'lambda_temperature': 0.8     # Moderate comfort penalty
    }
}

# Create environment
env = gym.make('Eplus-5zone-hot-continuous-v1', **env_params)

# Print configuration
print(f"Episode length: {env.get_wrapper_attr('timestep_per_episode')} timesteps")
print(f"Runperiod: {env.get_wrapper_attr('runperiod')}")

# Train PPO agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)
```

## 5. Best Practices

### For Runperiod Customization
- **Start small**: Use shorter periods for testing
- **Consider seasons**: Different comfort requirements
- **Balance length**: Longer episodes = more data but slower training
- **Weather files**: Ensure weather data covers your runperiod

### For Reward Customization
- **Start balanced**: `energy_weight = 0.5`
- **Adjust gradually**: Small changes to see effects
- **Monitor both**: Energy consumption and comfort violations
- **Consider context**: Different buildings need different priorities

### For Vectorization
- **Use DummyVecEnv** for single environment
- **Manual parallelization** for multiple environments
- **Monitor resources** when running multiple simulations
- **Consider alternatives** like ensemble methods

## Summary

**Vectorized Environments**: Limited support, primarily `DummyVecEnv`
**Runperiod Customization**: Full support with flexible time periods
**Reward Customization**: Extensive options with multiple reward functions

The key is to start with simple configurations and gradually customize based on your specific building and training requirements.