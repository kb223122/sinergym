# 5Zone Environment Customization Implementation Guide

## Overview

This guide shows you exactly how to customize reward weights and run periods in the Sinergym `Eplus-5zone-hot-continuous-v1` environment. You can run and verify these changes to confirm they work as expected.

## Key Customization Parameters

### Reward Weights
- **`energy_weight`**: Controls balance between energy and comfort (0.0 to 1.0)
  - 0.0 = All comfort, no energy consideration
  - 0.5 = Balanced (default)
  - 1.0 = All energy, no comfort consideration

- **`lambda_energy`**: Scales energy consumption penalty
  - Higher values = Stronger energy penalty
  - Default: 1e-4

- **`lambda_temperature`**: Scales comfort violation penalty
  - Higher values = Stronger comfort penalty
  - Default: 1.0

### Run Periods
- **`timestep_per_hour`**: Controls simulation granularity
  - 1 = 1-hour timesteps (default)
  - 2 = 2-hour timesteps
  - 4 = 4-hour timesteps

- **`runperiod`**: Controls episode duration
  - Format: (start_month, start_day, end_month, end_day)
  - Default: (1, 1, 12, 31) = Full year

## Implementation Examples

### 1. Energy-Focused Configuration

```python
import gymnasium as gym
import sinergym
from sinergym.utils.rewards import LinearReward

# Create environment
env = gym.make('Eplus-5zone-hot-continuous-v1')

# Custom reward function - Energy-focused
custom_reward = LinearReward(
    temperature_variables=['air_temperature'],
    energy_variables=['HVAC_electricity_demand_rate'],
    range_comfort_winter=(20.0, 23.5),
    range_comfort_summer=(23.0, 26.0),
    energy_weight=0.8,        # 80% weight on energy
    lambda_energy=2e-4,       # Higher energy penalty
    lambda_temperature=0.5     # Lower comfort penalty
)

# Apply custom reward
env.reward_fn = custom_reward

# Custom run period - 2-hour timesteps
env.timestep_per_hour = 2
env.runperiod = (1, 1, 12, 31)  # Full year

# Verify configuration
print(f"Energy Weight: {env.reward_fn.energy_weight}")
print(f"Lambda Energy: {env.reward_fn.lambda_energy}")
print(f"Lambda Temperature: {env.reward_fn.lambda_temperature}")
print(f"Timesteps per Hour: {env.timestep_per_hour}")
print(f"Run Period: {env.runperiod}")
```

### 2. Comfort-Focused Configuration

```python
# Comfort-focused reward function
comfort_reward = LinearReward(
    temperature_variables=['air_temperature'],
    energy_variables=['HVAC_electricity_demand_rate'],
    range_comfort_winter=(20.0, 23.5),
    range_comfort_summer=(23.0, 26.0),
    energy_weight=0.2,        # 20% weight on energy
    lambda_energy=5e-5,       # Lower energy penalty
    lambda_temperature=2.0     # Higher comfort penalty
)

env.reward_fn = comfort_reward

# Summer-only run period
env.timestep_per_hour = 1
env.runperiod = (6, 1, 8, 31)  # June 1 to August 31
```

### 3. Fast Training Configuration

```python
# Balanced reward with coarse timesteps for faster training
fast_reward = LinearReward(
    temperature_variables=['air_temperature'],
    energy_variables=['HVAC_electricity_demand_rate'],
    range_comfort_winter=(20.0, 23.5),
    range_comfort_summer=(23.0, 26.0),
    energy_weight=0.5,        # Balanced
    lambda_energy=1e-4,       # Default
    lambda_temperature=1.0     # Default
)

env.reward_fn = fast_reward

# 4-hour timesteps for faster simulation
env.timestep_per_hour = 4
env.runperiod = (1, 1, 12, 31)  # Full year
```

## Verification Methods

### 1. Reward Function Verification

```python
# Test reward calculation
test_obs = {
    'air_temperature': 28.0,  # Too hot
    'HVAC_electricity_demand_rate': 8000.0,  # High energy
    'month': 7, 'day_of_month': 15, 'hour': 14
}

reward, terms = env.reward_fn(test_obs)
print(f"Reward: {reward}")
print(f"Energy Term: {terms.get('energy_term', 'N/A')}")
print(f"Temperature Term: {terms.get('temperature_term', 'N/A')}")
```

### 2. Environment Configuration Verification

```python
# Verify environment settings
print(f"Timestep per Hour: {env.timestep_per_hour}")
print(f"Run Period: {env.runperiod}")
print(f"Timesteps per Episode: {env.timestep_per_episode}")

# Verify reward function parameters
if hasattr(env, 'reward_fn') and env.reward_fn is not None:
    print(f"Energy Weight: {env.reward_fn.energy_weight}")
    print(f"Lambda Energy: {env.reward_fn.lambda_energy}")
    print(f"Lambda Temperature: {env.reward_fn.lambda_temperature}")
```

## Common Configuration Patterns

### Energy Optimization Research
```python
reward_config = {
    'energy_weight': 0.8,
    'lambda_energy': 2e-4,
    'lambda_temperature': 0.5
}
run_period_config = {
    'timestep_per_hour': 1,
    'runperiod': (1, 1, 12, 31)
}
```

### Comfort Optimization Research
```python
reward_config = {
    'energy_weight': 0.2,
    'lambda_energy': 5e-5,
    'lambda_temperature': 2.0
}
run_period_config = {
    'timestep_per_hour': 1,
    'runperiod': (6, 1, 8, 31)  # Summer only
}
```

### Seasonal Analysis
```python
# Winter analysis
winter_config = {
    'timestep_per_hour': 1,
    'runperiod': (12, 1, 2, 28)
}

# Summer analysis
summer_config = {
    'timestep_per_hour': 1,
    'runperiod': (6, 1, 8, 31)
}
```

## Testing Your Customizations

### Quick Test Script

```python
#!/usr/bin/env python3
"""
Quick test to verify your customizations are working
"""

import gymnasium as gym
import sinergym
from sinergym.utils.rewards import LinearReward

def test_customization():
    # Create environment
    env = gym.make('Eplus-5zone-hot-continuous-v1')
    
    # Apply your custom reward
    custom_reward = LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(20.0, 23.5),
        range_comfort_summer=(23.0, 26.0),
        energy_weight=0.8,        # Your custom value
        lambda_energy=2e-4,       # Your custom value
        lambda_temperature=0.5     # Your custom value
    )
    env.reward_fn = custom_reward
    
    # Apply your custom run period
    env.timestep_per_hour = 2    # Your custom value
    env.runperiod = (1, 1, 12, 31)  # Your custom value
    
    # Verify
    print("✅ Customization Applied:")
    print(f"   Energy Weight: {env.reward_fn.energy_weight}")
    print(f"   Lambda Energy: {env.reward_fn.lambda_energy}")
    print(f"   Lambda Temperature: {env.reward_fn.lambda_temperature}")
    print(f"   Timesteps per Hour: {env.timestep_per_hour}")
    print(f"   Run Period: {env.runperiod}")
    
    # Test reward calculation
    test_obs = {
        'air_temperature': 28.0,
        'HVAC_electricity_demand_rate': 8000.0,
        'month': 7, 'day_of_month': 15, 'hour': 14
    }
    
    reward, terms = env.reward_fn(test_obs)
    print(f"   Test Reward: {reward:.4f}")
    
    env.close()

if __name__ == "__main__":
    test_customization()
```

## Expected Results

### Reward Weight Effects
- **Higher `energy_weight`**: More penalty for energy usage
- **Higher `lambda_energy`**: Stronger energy penalty scaling
- **Higher `lambda_temperature`**: Stronger comfort violation penalty
- **Lower rewards** (more negative) indicate worse performance

### Run Period Effects
- **Higher `timestep_per_hour`**: Faster simulation, less granular control
- **Shorter `runperiod`**: Faster training, less comprehensive analysis
- **Seasonal periods**: Focused analysis on specific climate conditions

## Best Practices

1. **Start with balanced settings**: Use `energy_weight=0.5` initially
2. **Test different configurations**: Experiment with various lambda values
3. **Monitor both metrics**: Track both energy consumption and comfort violations
4. **Verify configurations**: Always check that your changes are applied correctly
5. **Consider your use case**: Choose timesteps and periods based on your research goals

## Troubleshooting

### Common Issues
1. **Environment creation fails**: Ensure EnergyPlus is installed
2. **Reward calculation errors**: Check that all required observation keys are present
3. **Configuration not applied**: Verify that you're setting the environment attributes correctly

### Verification Checklist
- [ ] Environment creates successfully
- [ ] Custom reward function is applied
- [ ] Run period settings are correct
- [ ] Reward calculation works with test observations
- [ ] Episode length matches expected timesteps

This guide provides everything you need to customize the 5Zone environment for your specific research needs. The key is to experiment with different configurations and verify that your changes are working as expected.