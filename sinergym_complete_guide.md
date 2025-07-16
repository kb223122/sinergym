# Sinergym Complete Guide: From Scratch to Advanced

## Table of Contents
1. [What is Sinergym?](#what-is-sinergym)
2. [Repository Structure](#repository-structure)
3. [Basic Setup and Installation](#basic-setup-and-installation)
4. [Understanding Environments](#understanding-environments)
5. [Basic Usage Examples](#basic-usage-examples)
6. [Environment Configuration](#environment-configuration)
7. [Wrappers and Utilities](#wrappers-and-utilities)
8. [Reinforcement Learning Integration](#reinforcement-learning-integration)
9. [Logging and Monitoring](#logging-and-monitoring)
10. [Advanced Features](#advanced-features)
11. [Practical Examples](#practical-examples)

## What is Sinergym?

**Sinergym** is a Gymnasium-based interface for building energy simulation and optimization using reinforcement learning. It provides:

- **EnergyPlus Integration**: Connects to EnergyPlus simulation engine for realistic building energy modeling
- **Gymnasium Interface**: Standard RL environment interface compatible with any RL library
- **Multiple Building Types**: Various pre-configured building models (offices, data centers, warehouses, etc.)
- **Weather Variability**: Support for different weather conditions and stochastic weather patterns
- **Customizable Rewards**: Multiple reward functions for energy efficiency and comfort optimization
- **Stable Baselines 3 Integration**: Seamless integration with popular RL algorithms
- **Logging and Monitoring**: Comprehensive logging with CSV and Weights & Biases support

## Repository Structure

```
sinergym/
├── sinergym/                    # Main source code
│   ├── envs/                   # Environment implementations
│   ├── simulators/             # EnergyPlus simulator
│   ├── utils/                  # Utilities, wrappers, rewards
│   ├── config/                 # Building model configuration
│   └── data/                   # Default configurations and building files
├── examples/                   # Jupyter notebook examples
├── scripts/                    # Training and evaluation scripts
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

## Basic Setup and Installation

Since you mentioned everything is already installed and running, let's verify the setup:

```python
import gymnasium as gym
import sinergym
import numpy as np

# Check available environments
print("Available environments:")
for env_id in gym.envs.registration.registry.keys():
    if env_id.startswith('Eplus'):
        print(f"  - {env_id}")
```

## Understanding Environments

### Environment Types

Sinergym provides several types of environments:

1. **Demo Environment**: `Eplus-demo-v1` - Simple 5-zone building for testing
2. **5-Zone Building**: Various configurations (hot, mixed, cool climates)
3. **Data Center**: Specialized environments for data center optimization
4. **Office Buildings**: ASHRAE standard office buildings
5. **Warehouses**: Industrial building optimization
6. **Residential**: Radiant heating/cooling systems

### Environment Naming Convention

Environment names follow this pattern:
```
Eplus-{building_type}-{climate}-{action_space}-{weather_type}-v{version}
```

Examples:
- `Eplus-5zone-hot-continuous-stochastic-v1`
- `Eplus-datacenter-mixed-discrete-v1`
- `Eplus-ASHRAE901_OfficeMedium_STD2019_Denver-continuous-v1`

## Basic Usage Examples

### 1. Simple Environment Interaction

```python
import gymnasium as gym
import sinergym

# Create environment
env = gym.make('Eplus-demo-v1')

# Reset environment
obs, info = env.reset()
print(f"Initial observation: {obs}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Run one episode
rewards = []
terminated = truncated = False

while not (terminated or truncated):
    # Take random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    
    # Print info every 100 steps
    if len(rewards) % 100 == 0:
        print(f"Step {len(rewards)}, Reward: {reward:.2f}")

print(f"Episode finished. Total reward: {sum(rewards):.2f}")
env.close()
```

### 2. Understanding Observations and Actions

```python
import gymnasium as gym
import sinergym

env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')

# Get environment information
print("Environment Information:")
print(f"Name: {env.name}")
print(f"Episode length: {env.episode_length} seconds")
print(f"Timesteps per episode: {env.timestep_per_episode}")
print(f"Step size: {env.step_size} seconds")

# Reset and examine observation
obs, info = env.reset()
print(f"\nObservation shape: {obs.shape}")
print(f"Observation variables: {env.observation_variables}")
print(f"Action variables: {env.action_variables}")

# Print first observation
print(f"\nFirst observation: {obs}")

env.close()
```

## Environment Configuration

### Understanding Configuration Files

Environment configurations are stored in YAML files in `sinergym/data/default_configuration/`. Let's examine a configuration:

```python
import yaml

# Read configuration file
with open('sinergym/data/default_configuration/5ZoneAutoDXVAV.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Configuration structure:")
for key, value in config.items():
    print(f"  {key}: {type(value).__name__}")
```

### Key Configuration Components

1. **Building File**: The EnergyPlus building model (`.epJSON` format)
2. **Weather Files**: Climate data files (`.epw` format)
3. **Variables**: EnergyPlus output variables to observe
4. **Actuators**: Controllable building systems
5. **Reward Function**: How to calculate rewards
6. **Action Space**: Available control actions

### Custom Environment Creation

```python
import gymnasium as gym
import numpy as np
from sinergym.utils.rewards import LinearReward

# Create custom environment
env = gym.make('Eplus-demo-v1',
               env_name='my-custom-env',
               config_params={
                   'runperiod': (1, 1, 1991, 1, 31, 1991),  # January 1991
                   'timesteps_per_hour': 4  # 15-minute timesteps
               })

print(f"Custom environment created: {env.name}")
env.close()
```

## Wrappers and Utilities

### Available Wrappers

Sinergym provides several wrappers to enhance environment functionality:

```python
from sinergym.utils.wrappers import (
    NormalizeAction,
    NormalizeObservation,
    LoggerWrapper,
    CSVLogger,
    WandBLogger,
    DiscretizeEnv
)

# Create environment with wrappers
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')

# Normalize observations and actions (recommended for RL)
env = NormalizeObservation(env)
env = NormalizeAction(env)

# Add logging
env = LoggerWrapper(env)
env = CSVLogger(env)

# Optional: Weights & Biases logging
# env = WandBLogger(env, entity='your-entity', project_name='your-project')

print("Environment wrapped with normalization and logging")
```

### Wrapper Benefits

1. **NormalizeAction**: Scales actions to [-1, 1] range for better RL training
2. **NormalizeObservation**: Normalizes observations for stable training
3. **LoggerWrapper**: Provides detailed logging capabilities
4. **CSVLogger**: Saves episode data to CSV files
5. **WandBLogger**: Integrates with Weights & Biases for experiment tracking

## Reinforcement Learning Integration

### Training with Stable Baselines 3

```python
import gymnasium as gym
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sinergym.utils.wrappers import NormalizeAction, NormalizeObservation, LoggerWrapper, CSVLogger

# Create training environment
train_env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
train_env = NormalizeObservation(train_env)
train_env = NormalizeAction(train_env)
train_env = LoggerWrapper(train_env)
train_env = CSVLogger(train_env)

# Create evaluation environment
eval_env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)

# Create model
model = PPO('MlpPolicy', train_env, verbose=1)

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Train the model
model.learn(total_timesteps=100000, callback=eval_callback)

# Save the model
model.save("ppo_sinergym")

# Test the trained model
obs, info = eval_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()
train_env.close()
```

### Using Training Scripts

Sinergym provides pre-configured training scripts:

```bash
# Train with PPO
python scripts/train/train_agent_local_conf.py \
    --config scripts/train/local_confs/conf_examples/train_agent_PPO.yaml

# Train with SAC
python scripts/train/train_agent_local_conf.py \
    --config scripts/train/local_confs/conf_examples/train_agent_SAC.yaml
```

## Logging and Monitoring

### CSV Logging

```python
import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import CSVLogger

env = gym.make('Eplus-demo-v1')
env = CSVLogger(env)

# Run episode
obs, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()

# Check generated CSV files
import os
print("Generated CSV files:")
for file in os.listdir(env.episode_path):
    if file.endswith('.csv'):
        print(f"  - {file}")
```

### Weights & Biases Integration

```python
import os
import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import WandBLogger

# Set your WANDB API key
os.environ['WANDB_API_KEY'] = 'your-api-key'

env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
env = WandBLogger(
    env,
    entity='your-entity',
    project_name='sinergym-experiments',
    run_name='test-run',
    tags=['test', '5zone']
)

# Run episode (data will be logged to W&B)
obs, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Advanced Features

### Weather Variability

```python
import gymnasium as gym
import sinergym

# Environment with weather variability
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')

# Weather variability adds noise to temperature data
# This makes the environment more realistic and challenging

obs, info = env.reset()
print(f"Weather variability enabled: {env.weather_variability is not None}")

env.close()
```

### Custom Reward Functions

```python
import gymnasium as gym
import sinergym
from sinergym.utils.rewards import LinearReward, ExpReward

# Environment with custom reward
env = gym.make('Eplus-demo-v1',
               reward=ExpReward,
               reward_kwargs={
                   'temperature_variables': ['air_temperature'],
                   'energy_variables': ['HVAC_electricity_demand_rate'],
                   'range_comfort_winter': (20.0, 23.5),
                   'range_comfort_summer': (23.0, 26.0),
                   'energy_weight': 0.5
               })

print("Custom reward function applied")
env.close()
```

### Context Variables

```python
import gymnasium as gym
import sinergym

# Environment with context variables (building parameters that can be changed)
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')

# Update context during simulation
obs, info = env.reset()
context_values = [0.5, 0.3]  # Example context values
env.update_context(context_values)

print("Context variables updated")
env.close()
```

## Practical Examples

### Example 1: Energy Optimization

```python
import gymnasium as gym
import sinergym
import numpy as np

# Create environment for energy optimization
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')

# Simple rule-based controller
def rule_based_controller(obs):
    """Simple rule-based controller for energy optimization"""
    outdoor_temp = obs[0]  # Assuming first observation is outdoor temperature
    indoor_temp = obs[8]   # Assuming 9th observation is indoor temperature
    
    # Heating setpoint
    if outdoor_temp < 15:
        heating_setpoint = 22.0
    else:
        heating_setpoint = 20.0
    
    # Cooling setpoint
    if outdoor_temp > 25:
        cooling_setpoint = 24.0
    else:
        cooling_setpoint = 26.0
    
    return np.array([heating_setpoint, cooling_setpoint], dtype=np.float32)

# Run episode with rule-based controller
obs, info = env.reset()
rewards = []
energy_consumption = []

terminated = truncated = False
while not (terminated or truncated):
    action = rule_based_controller(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    energy_consumption.append(info.get('total_power_demand', 0))

print(f"Total energy consumption: {sum(energy_consumption):.2f} kWh")
print(f"Average reward: {np.mean(rewards):.2f}")

env.close()
```

### Example 2: Comfort vs Energy Trade-off

```python
import gymnasium as gym
import sinergym
import numpy as np

# Create environment with comfort-focused reward
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1',
               reward_kwargs={
                   'energy_weight': 0.3,  # Lower energy weight
                   'lambda_temperature': 2.0  # Higher comfort penalty
               })

# Comfort-focused controller
def comfort_controller(obs):
    """Controller that prioritizes comfort over energy"""
    outdoor_temp = obs[0]
    
    # Always maintain comfortable temperature range
    heating_setpoint = 21.0
    cooling_setpoint = 25.0
    
    return np.array([heating_setpoint, cooling_setpoint], dtype=np.float32)

# Run episode
obs, info = env.reset()
rewards = []
comfort_violations = []

terminated = truncated = False
while not (terminated or truncated):
    action = comfort_controller(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    comfort_violations.append(info.get('total_temperature_violation', 0))

print(f"Comfort violations: {sum(comfort_violations):.2f}")
print(f"Average reward: {np.mean(rewards):.2f}")

env.close()
```

### Example 3: Multi-Episode Training

```python
import gymnasium as gym
import sinergym
from stable_baselines3 import PPO
from sinergym.utils.wrappers import NormalizeAction, NormalizeObservation, LoggerWrapper, CSVLogger

# Create environment
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
env = NormalizeObservation(env)
env = NormalizeAction(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

# Create model
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003)

# Train for multiple episodes
total_episodes = 10
for episode in range(total_episodes):
    print(f"Training episode {episode + 1}/{total_episodes}")
    
    # Train for one episode
    model.learn(total_timesteps=env.timestep_per_episode)
    
    # Evaluate
    obs, info = env.reset()
    episode_reward = 0
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
    
    print(f"Episode {episode + 1} reward: {episode_reward:.2f}")

# Save final model
model.save("multi_episode_ppo")

env.close()
```

## Key Takeaways

1. **Environment Creation**: Use `gym.make()` with environment IDs
2. **Wrappers**: Always use normalization wrappers for RL training
3. **Logging**: Use CSVLogger and WandBLogger for experiment tracking
4. **Rewards**: Understand the reward function and its parameters
5. **Weather**: Consider weather variability for realistic scenarios
6. **Training**: Use Stable Baselines 3 for RL training
7. **Evaluation**: Always evaluate on separate environments

## Next Steps

1. **Explore Examples**: Run through the Jupyter notebooks in the `examples/` directory
2. **Custom Environments**: Create your own building configurations
3. **Advanced RL**: Experiment with different algorithms and hyperparameters
4. **Real Buildings**: Apply to real building optimization problems
5. **Research**: Contribute to the Sinergym community

This guide covers the fundamentals to get you started with Sinergym. The repository is well-documented and has extensive examples to help you explore advanced features.