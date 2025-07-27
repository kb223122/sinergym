# 5Zone Hot Environment Customization Guide

This guide shows you how to customize the `Eplus-5zone-hot-continuous-v1` environment in Sinergym with different reward parameters, run periods, and configurations.

## 🚀 Quick Start

### 1. Run the Comprehensive Guide
```bash
python comprehensive_5zone_customization_guide.py
```
This will show you ALL possible customizations and run a practical training example.

### 2. Test Reward Parameters
```bash
python test_reward_parameters.py
```
This will test different reward parameter combinations and show how they affect rewards.

### 3. Use Custom Configuration
```bash
python scripts/train/local_confs/train_agent_local_conf.py -conf custom_5zone_config.yaml
```
This will train a PPO agent with your custom configuration.

## 📁 Files Overview

### Main Files:
- **`comprehensive_5zone_customization_guide.py`** - Complete demonstration of all customizations
- **`test_reward_parameters.py`** - Test different reward parameters
- **`custom_5zone_config.yaml`** - YAML configuration for custom training

### What Each File Does:

#### 1. `comprehensive_5zone_customization_guide.py`
- **Environment Analysis**: Shows detailed info about the 5zone environment
- **Reward Function Testing**: Tests 6 different reward functions
- **Custom Environments**: Creates environments with different run periods
- **Practical Training**: Trains a PPO agent with custom settings
- **Model Evaluation**: Evaluates the trained model

#### 2. `test_reward_parameters.py`
- **Parameter Testing**: Tests 5 different reward parameter combinations
- **Reward Calculation**: Shows how rewards change with different parameters
- **Custom Environment**: Creates an environment with custom reward function

#### 3. `custom_5zone_config.yaml`
- **YAML Configuration**: Ready-to-use configuration file
- **Custom Run Period**: 1 week instead of 1 year
- **Custom Reward**: Energy-focused reward function
- **Weather Variability**: Adds realistic weather noise

## 🎯 Key Customizations Explained

### 1. Reward Parameters

#### Energy Weight (`energy_weight`)
- **Range**: 0.0 to 1.0
- **Default**: 0.5 (balanced)
- **Higher values**: More focus on energy efficiency
- **Lower values**: More focus on thermal comfort

#### Lambda Energy (`lambda_energy`)
- **Range**: 1e-5 to 1e-3
- **Default**: 1.0e-4
- **Higher values**: Stronger penalty for high energy consumption
- **Lower values**: Less penalty for energy consumption

#### Lambda Temperature (`lambda_temperature`)
- **Range**: 0.1 to 5.0
- **Default**: 1.0
- **Higher values**: Stronger penalty for comfort violations
- **Lower values**: Less penalty for comfort violations

### 2. Run Period Customization

#### Default (1 Year)
```python
config_params={
    'runperiod': (1, 1, 1991, 12, 31, 1991),  # Full year
    'timesteps_per_hour': 1  # 1-hour timesteps
}
```

#### 1 Month
```python
config_params={
    'runperiod': (1, 1, 1991, 1, 31, 1991),  # January only
    'timesteps_per_hour': 1  # 1-hour timesteps
}
```

#### 1 Week
```python
config_params={
    'runperiod': (1, 1, 1991, 1, 7, 1991),  # First week
    'timesteps_per_hour': 4  # 15-minute timesteps
}
```

#### 1 Day
```python
config_params={
    'runperiod': (1, 1, 1991, 1, 1, 1991),  # Single day
    'timesteps_per_hour': 12  # 5-minute timesteps
}
```

### 3. Timestep Customization

#### Available Options:
- **`timesteps_per_hour: 1`** → 1 timestep = 1 hour
- **`timesteps_per_hour: 2`** → 1 timestep = 30 minutes
- **`timesteps_per_hour: 4`** → 1 timestep = 15 minutes
- **`timesteps_per_hour: 6`** → 1 timestep = 10 minutes
- **`timesteps_per_hour: 12`** → 1 timestep = 5 minutes

### 4. Weather Variability

Add realistic noise to weather data:
```python
weather_variability={
    'Site Outdoor Air DryBulb Temperature': (2.0, 0.0, 24.0),  # sigma, mu, tau
    'Site Outdoor Air Relative Humidity': (5.0, 0.0, 24.0)
}
```

## 🎯 Reward Functions Available

### 1. LinearReward (Default)
```python
LinearReward(
    energy_weight=0.5,
    lambda_energy=1.0e-4,
    lambda_temperature=1.0
)
```

### 2. ExpReward (Exponential Penalties)
```python
ExpReward(
    energy_weight=0.5,
    lambda_energy=1.0e-4,
    lambda_temperature=1.0
)
```

### 3. HourlyLinearReward (Time-based)
```python
HourlyLinearReward(
    energy_weight=0.5,
    lambda_energy=1.0e-4,
    lambda_temperature=1.0,
    range_comfort_hours=(9, 19)  # 9 AM to 7 PM
)
```

### 4. NormalizedLinearReward (Normalized Penalties)
```python
NormalizedLinearReward(
    energy_weight=0.5,
    max_energy_penalty=8,
    max_comfort_penalty=12
)
```

## 🔧 Environment Information

### Default 5Zone Hot Environment:
- **Building**: 5-zone commercial building with VAV HVAC
- **Weather**: Hot climate (Arizona)
- **Action Space**: 2 continuous variables (heating/cooling setpoints)
- **Observation Space**: 15 variables (temperature, energy, weather, etc.)
- **Episode Length**: 35,040 timesteps (1 year)
- **Timestep**: 15 minutes (900 seconds)

### Action Space:
- **Action 1**: Heating setpoint (12.0°C to 23.25°C)
- **Action 2**: Cooling setpoint (23.25°C to 30.0°C)

### Observation Variables:
- Time: month, day_of_month, hour
- Weather: outdoor temperature, humidity, wind, solar radiation
- Building: zone temperature, humidity, occupancy
- Energy: HVAC electricity demand, CO2 emissions
- Setpoints: current heating/cooling setpoints

## 📊 Training Configuration

### PPO Parameters (Recommended):
```yaml
algorithm:
  name: PPO
  parameters:
    learning_rate: 0.0003
    n_steps: 2048  # For 1-year episodes
    n_steps: 1024  # For shorter episodes
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01
    vf_coef: 0.5
```

### Wrappers (Recommended):
```python
env = NormalizeObservation(env)  # Normalize observations
env = NormalizeAction(env)       # Normalize actions
env = LoggerWrapper(env)         # Log interactions
env = CSVLogger(env)            # Save to CSV
```

## 🎯 Practical Examples

### Example 1: Energy-Focused Training
```python
# Create energy-focused reward
energy_reward = LinearReward(
    energy_weight=0.8,        # 80% energy focus
    lambda_energy=2.0e-4,     # High energy penalty
    lambda_temperature=0.5    # Low comfort penalty
)

# Create environment
env = gym.make(
    'Eplus-5zone-hot-continuous-v1',
    reward=energy_reward,
    config_params={
        'runperiod': (1, 1, 1991, 1, 7, 1991),  # 1 week
        'timesteps_per_hour': 4  # 15-minute timesteps
    }
)
```

### Example 2: Comfort-Focused Training
```python
# Create comfort-focused reward
comfort_reward = LinearReward(
    energy_weight=0.2,        # 20% energy focus
    lambda_energy=0.5e-4,     # Low energy penalty
    lambda_temperature=2.0    # High comfort penalty
)

# Create environment
env = gym.make(
    'Eplus-5zone-hot-continuous-v1',
    reward=comfort_reward
)
```

### Example 3: Short Training for Testing
```python
# Create environment for quick testing
env = gym.make(
    'Eplus-5zone-hot-continuous-v1',
    config_params={
        'runperiod': (1, 1, 1991, 1, 1, 1991),  # 1 day
        'timesteps_per_hour': 12  # 5-minute timesteps
    }
)
```

## 📈 Monitoring and Evaluation

### Key Metrics to Monitor:
- **Episode Reward**: Total reward per episode
- **Energy Consumption**: HVAC electricity demand
- **Comfort Violations**: Time outside comfort range
- **Average Temperature**: Zone air temperature
- **Action Values**: Heating/cooling setpoints

### Evaluation Script:
```python
# Evaluate trained model
model = PPO.load('trained_model.zip')
obs, info = env.reset()

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: Reward={reward:.4f}, Temp={info['air_temperature']:.1f}°C")
```

## 🔧 Troubleshooting

### Common Issues:

1. **EnergyPlus not found**
   - Install EnergyPlus 24.2.0
   - Set ENERGYPLUS_INSTALL_DIR environment variable

2. **Memory issues**
   - Reduce batch_size or n_steps
   - Use shorter run periods for testing

3. **Slow training**
   - Use shorter episodes for initial testing
   - Reduce timesteps_per_hour
   - Use GPU if available

4. **Import errors**
   - Install with DRL extras: `pip install sinergym[drl]`
   - Check Python version (3.12 recommended)

### Performance Tips:
- Start with short episodes (1 day or 1 week) for testing
- Use 15-minute timesteps for good balance of speed and accuracy
- Normalize observations and actions for better training
- Monitor training progress with CSVLogger or WandBLogger

## 🎉 Next Steps

After running these examples, you can:

1. **Experiment with different reward parameters**
2. **Try different run periods and timesteps**
3. **Test different reward functions**
4. **Add weather variability**
5. **Train for longer periods**
6. **Compare different algorithms (SAC, TD3, etc.)**

## 📚 Additional Resources

- [Sinergym Documentation](https://ugr-sail.github.io/sinergym/)
- [Stable Baselines 3 Documentation](https://stable-baselines3.readthedocs.io/)
- [EnergyPlus Documentation](https://energyplus.net/documentation)

Happy training! 🚀