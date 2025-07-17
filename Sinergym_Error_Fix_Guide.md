# Sinergym Environment Configuration Error Fix Guide

## The Problem

You encountered this error when trying to create a Sinergym environment:

```
TypeError: EplusEnv.__init__() got an unexpected keyword argument 'config_params'
```

This error occurs because youre trying to pass `config_params` and `env_name` directly to `gym.make()`, but `gym.make()` doesn't accept these parameters as additional keyword arguments.

## Root Cause Analysis

### How Sinergym Environments Work1nment Registration**: Sinergym environments are registered in `sinergym/__init__.py` using `gym.register()`
2onfiguration Files**: Environment configurations are stored in YAML files in `sinergym/data/default_configuration/`
3. **Parameter Processing**: The `convert_conf_to_env_parameters()` function converts YAML configurations to environment constructor parameters
4. **Environment Creation**: When you call `gym.make()`, it uses the registered parameters to create the environment

### The Issue in Your Code

**Original (Incorrect) Code:**
```python
train_env = gym.make(env_id, config_params=extra_conf, env_name=experiment_name)
```

**Problem**: You're passing `config_params` and `env_name` as additional keyword arguments to `gym.make()`, but these should be part of the environment's configuration, not passed separately.

## The Solution

### Method 1: Use Pre-configured Environments (Recommended)

The easiest solution is to use the pre-configured environments that already have the settings you need:

```python
# Use the pre-configured environment
env_id = 'Eplus-5zone-hot-continuous-v1ain_env = gym.make(env_id)

# Apply wrappers
train_env = LoggerWrapper(train_env)
train_env = CSVLogger(train_env)
train_env = NormalizeObservation(train_env)
train_env = NormalizeAction(train_env)

# Set custom reward function if needed
reward_kwargs = [object Object]temperature_variables": ["air_temperature"],
   energy_variables:["HVAC_electricity_demand_rate"],
  range_comfort_winter":20 23.5    range_comfort_summer":23026
    summer_start": [61
    summer_final": [9, 30],
    energy_weight":0.4    lambda_energy":00.01lambda_temperature: 28}

train_env.set_wrapper_attr('reward_fn, LinearReward(**reward_kwargs))
```

### Method 2: Create Custom Environment Configuration

If you need custom configuration parameters, create a custom environment:

```python
import gymnasium as gym
import sinergym
from sinergym.envs import EplusEnv

# Define your custom configuration
custom_config = {
    building_file: 5ZoneAutoDXVAV.epJSON',
    weather_files: USA_AZ_Davis-Monthan.AFB.722745.epw',
 action_space': gym.spaces.Box(
        low=np.array([12, 23.25, dtype=np.float32),
        high=np.array([23.250, dtype=np.float32),
        shape=(2,
        dtype=np.float32    ),
   time_variables': ['month,day_of_month', 'hour],
 variables': [object Object]outdoor_temperature': ('Site Outdoor Air DryBulb Temperature', 'Environment'),
       outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment),  air_temperature': ('Zone Air Temperature', 'SPACE51),
       HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate,Whole Building)
    },
 actuators': [object Object]      Heating_Setpoint_RL': ('Schedule:Compact,Schedule Value',HTG-SETP-SCH'),
        Cooling_Setpoint_RL': ('Schedule:Compact,Schedule Value', 'CLG-SETP-SCH')
    },
  reward': LinearReward,
  reward_kwargs:reward_kwargs,
    'env_name: experiment_name,
  config_params':[object Object]      timesteps_per_hour: 1,        runperiod: (1,1 199131, 12, 1991)
    }
}

# Create environment directly
train_env = EplusEnv(**custom_config)
```

### Method 3: Modify Environment After Creation

You can also modify some parameters after creating the environment:

```python
# Create basic environment
train_env = gym.make('Eplus-5zone-hot-continuous-v1')

# Modify reward function
train_env.set_wrapper_attr('reward_fn, LinearReward(**reward_kwargs))

# Note: Some parameters like runperiod and timesteps_per_hour 
# cannot be changed after environment creation
```

## Available Environment Configurations

Sinergym provides several pre-configured environments:

### 5-Zone Building Environments
- `Eplus-5zone-hot-continuous-v1` - Hot climate, continuous actions
- `Eplus-5zone-mixed-continuous-v1` - Mixed climate, continuous actions  
- `Eplus-5zone-cool-continuous-v1limate, continuous actions
- `Eplus-5zone-hot-continuous-stochastic-v1` - Hot climate with weather variability

### Other Building Types
- `Eplus-demo-v1` - Simple demo environment
- `Eplus-datacenter-mixed-discrete-v1` - Data center with discrete actions
- `Eplus-ASHRAE901fficeMedium_STD2019_Denver-continuous-v1AE office building

## Configuration Parameters Explained

### config_params
The `config_params` dictionary can contain:
- `timesteps_per_hour`: Number of simulation timesteps per hour (default:1eriod`: Simulation period as (start_month, start_day, start_year, end_month, end_day, end_year)
- `output_directory`: Custom output directory
- `epjson`: EnergyPlus JSON file path

### reward_kwargs
Reward function parameters:
- `temperature_variables`: List of temperature variables to monitor
- `energy_variables`: List of energy consumption variables
- `range_comfort_winter`: Comfort temperature range for winter [min, max]
- `range_comfort_summer`: Comfort temperature range for summer [min, max]
- `energy_weight`: Weight for energy consumption in reward calculation
- `lambda_energy`: Energy penalty coefficient
- `lambda_temperature`: Temperature penalty coefficient

## Best Practices1Use Pre-configured Environments**: Start with existing environments unless you need specific customizations
2fy Reward Function**: Use `set_wrapper_attr()` to change reward functions after environment creation
3. **Check Available Parameters**: Always check the environment's documentation for available configuration options
4. **Test Incrementally**: Test your environment configuration step by step to identify issues early

## Complete Fixed Example

Here's your complete corrected code:

```python
import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import (
    LoggerWrapper, CSVLogger, NormalizeObservation, NormalizeAction
)
from sinergym.utils.rewards import LinearReward
from stable_baselines3 import PPO

# Reward configuration
reward_kwargs = [object Object]temperature_variables": ["air_temperature"],
   energy_variables:["HVAC_electricity_demand_rate"],
  range_comfort_winter":20 23.5    range_comfort_summer":23026
    summer_start": [61
    summer_final": [9, 30],
    energy_weight":0.4    lambda_energy":00.01lambda_temperature": 28
}

# Create environment (FIXED: no config_params or env_name in gym.make())
env_id = 'Eplus-5zone-hot-continuous-v1ain_env = gym.make(env_id)

# Apply wrappers
train_env = LoggerWrapper(train_env)
train_env = CSVLogger(train_env)
train_env = NormalizeObservation(train_env)
train_env = NormalizeAction(train_env)

# Set custom reward function
train_env.set_wrapper_attr('reward_fn, LinearReward(**reward_kwargs))

# Create and train model
model = PPO(MlpPolicy', train_env, verbose=1)
model.learn(total_timesteps=10000)
```

## Additional Resources

- [Sinergym Documentation](https://ugr-sail.github.io/sinergym/)
- [Environment Configuration Guide](https://ugr-sail.github.io/sinergym/compilation/main/pages/environments_registration.html)
- [Available Environments](https://ugr-sail.github.io/sinergym/compilation/main/pages/environments.html)
-Reward Functions](https://ugr-sail.github.io/sinergym/compilation/main/pages/rewards.html)