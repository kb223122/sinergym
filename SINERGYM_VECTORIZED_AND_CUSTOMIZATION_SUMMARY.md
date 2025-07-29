# Sinergym Vectorized Environments and Customization: Direct Answers

## Question 1: Does Sinergym support vectorized environments like SubprocVecEnv?

### **Answer: Limited Support**

**❌ No SubprocVecEnv Support**
- Sinergym does **NOT** support `SubprocVecEnv` or other multiprocessing vectorized environments
- No examples found in the codebase for `SubprocVecEnv`
- EnergyPlus simulations are too resource-intensive for parallel processing

**✅ DummyVecEnv Support**
- Sinergym **DOES** support `DummyVecEnv` for single-environment vectorization
- Example from `reinforcement_learning_example.py`:
```python
from stable_baselines3.common.vec_env import DummyVecEnv

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

### Why No SubprocVecEnv?
1. **EnergyPlus Complexity**: Each environment runs full EnergyPlus simulation
2. **Resource Intensive**: Multiple EnergyPlus processes would be very heavy
3. **File System Conflicts**: Multiple processes writing to same directories
4. **Design Philosophy**: Sinergym focuses on single, detailed simulations

## Question 2: How to customize runperiod and reward weights?

### **Answer: Full Support with Extensive Options**

## Runperiod Customization

### **Format**: `(start_day, start_month, start_year, end_day, end_month, end_year)`

### **Examples**:

#### 1. Full Year Simulation
```python
env_params = {
    'runperiod': [1, 1, 1991, 31, 12, 1991],  # Jan 1 to Dec 31, 1991
    'timesteps_per_hour': 4
}
```

#### 2. Summer Season Only
```python
env_params = {
    'runperiod': [1, 6, 1991, 31, 8, 1991],  # June 1 to Aug 31
    'timesteps_per_hour': 4
}
```

#### 3. Winter Season Only
```python
env_params = {
    'runperiod': [1, 12, 1991, 28, 2, 1992],  # Dec 1 to Feb 28
    'timesteps_per_hour': 4
}
```

#### 4. Specific Month
```python
env_params = {
    'runperiod': [1, 7, 1991, 31, 7, 1991],  # July only
    'timesteps_per_hour': 4
}
```

## Reward Weight Customization

### **Available Reward Functions**:
1. **LinearReward** (most common)
2. **ExpReward** (exponential penalties)
3. **EnergyCostLinearReward** (includes energy costs)
4. **HourlyLinearReward** (time-dependent weights)
5. **NormalizedLinearReward** (normalized penalties)
6. **MultiZoneReward** (multi-zone comfort)

### **Basic Structure**:
```python
reward_config = {
    'temperature_variables': ['air_temperature'],
    'energy_variables': ['HVAC_electricity_demand_rate'],
    'range_comfort_winter': [20.0, 23.5],
    'range_comfort_summer': [23.0, 26.0],
    'summer_start': [6, 1],
    'summer_final': [9, 30],
    'energy_weight': 0.5,        # Controls energy vs comfort balance
    'lambda_energy': 0.0001,     # Energy penalty scaling
    'lambda_temperature': 1.0     # Comfort penalty scaling
}
```

### **Customization Examples**:

#### 1. Energy-Focused Training
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

#### 2. Comfort-Focused Training
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

#### 3. Balanced Training
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

### **Advanced Customization**:

#### Custom Comfort Ranges
```python
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
```

#### Multiple Energy Variables
```python
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
```

## Complete Working Example

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

## Configuration Comparison

| Configuration | Runperiod | Energy Weight | Comfort Penalty |
|---------------|-----------|---------------|-----------------|
| Summer Energy-Focused | 6/1-8/31 | 0.8 | 0.5 |
| Winter Comfort-Focused | 12/1-2/28 | 0.2 | 2.0 |
| Full Year Balanced | 1/1-12/31 | 0.5 | 1.0 |
| Custom Comfort Ranges | 3/1-5/31 | 0.6 | 1.5 |

## Summary

### **Vectorized Environments**:
- ❌ **No SubprocVecEnv support**
- ✅ **DummyVecEnv support available**
- 🔧 **Manual parallelization possible**

### **Customization**:
- ✅ **Full runperiod customization**
- ✅ **Extensive reward weight options**
- ✅ **Multiple reward function types**
- ✅ **Flexible comfort ranges**
- ✅ **Multiple energy variables**

The key is to start with simple configurations and gradually customize based on your specific building and training requirements.