# Complete Guide to 5Zone Hot Environment Customization

## 🎯 What You've Accomplished

You now have a complete understanding of how to customize the `Eplus-5zone-hot-continuous-v1` environment in Sinergym! Here's what you can do:

## 📁 Files Created for You

### 1. `simple_5zone_demo.py` ✅ (Ready to run)
- **What it does**: Demonstrates reward parameter effects without requiring Sinergym
- **How to run**: `python3 simple_5zone_demo.py`
- **What you'll see**: How different reward parameters affect the reward calculation

### 2. `comprehensive_5zone_customization_guide.py` (Requires Sinergym)
- **What it does**: Complete demonstration of ALL customizations
- **How to run**: `python comprehensive_5zone_customization_guide.py` (after installing Sinergym)
- **What you'll see**: Environment analysis, reward testing, custom environments, and PPO training

### 3. `test_reward_parameters.py` (Requires Sinergym)
- **What it does**: Tests different reward parameter combinations
- **How to run**: `python test_reward_parameters.py` (after installing Sinergym)
- **What you'll see**: Detailed comparison of reward functions

### 4. `custom_5zone_config.yaml` (Ready to use)
- **What it does**: YAML configuration file for custom training
- **How to use**: `python scripts/train/local_confs/train_agent_local_conf.py -conf custom_5zone_config.yaml`
- **What it contains**: Custom run period, reward parameters, weather variability

### 5. `README_5zone_customization.md` (Documentation)
- **What it does**: Complete documentation of all customizations
- **How to use**: Reference guide for all features

## 🚀 Quick Start Guide

### Step 1: Understand the Concepts (No Installation Required)
```bash
python3 simple_5zone_demo.py
```
This will show you how reward parameters work and give you a complete overview.

### Step 2: Install Sinergym (When Ready)
```bash
pip install sinergym[drl]
```

### Step 3: Run the Comprehensive Guide
```bash
python comprehensive_5zone_customization_guide.py
```

### Step 4: Use Custom Configuration
```bash
python scripts/train/local_confs/train_agent_local_conf.py -conf custom_5zone_config.yaml
```

## 🎯 Key Customizations You Can Make

### 1. Reward Parameters

#### Energy Weight (`energy_weight`)
- **Range**: 0.0 to 1.0
- **Default**: 0.5 (balanced)
- **Higher values**: More focus on energy efficiency
- **Lower values**: More focus on thermal comfort

**Example**:
```python
# Energy-focused reward
energy_reward = LinearReward(
    energy_weight=0.8,        # 80% energy focus
    lambda_energy=2.0e-4,     # High energy penalty
    lambda_temperature=0.5    # Low comfort penalty
)

# Comfort-focused reward
comfort_reward = LinearReward(
    energy_weight=0.2,        # 20% energy focus
    lambda_energy=0.5e-4,     # Low energy penalty
    lambda_temperature=2.0    # High comfort penalty
)
```

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

## 🎉 What You Can Do Now

### Immediate Actions:
1. **Run the demo**: `python3 simple_5zone_demo.py` (no installation needed)
2. **Understand concepts**: See how reward parameters affect calculations
3. **Plan experiments**: Decide what customizations you want to try

### After Installing Sinergym:
1. **Run comprehensive guide**: `python comprehensive_5zone_customization_guide.py`
2. **Test reward parameters**: `python test_reward_parameters.py`
3. **Train with custom config**: Use `custom_5zone_config.yaml`
4. **Experiment**: Try different reward functions and parameters

### Advanced Experiments:
1. **Compare algorithms**: Try SAC, TD3, DDPG
2. **Multi-environment training**: Train on multiple weather conditions
3. **Custom reward functions**: Create your own reward functions
4. **Hyperparameter tuning**: Optimize PPO parameters
5. **Real-world deployment**: Apply trained models to real buildings

## 📚 Key Takeaways

### Reward Function Formula:
```
R = -W * λ_E * Energy - (1-W) * λ_T * Temperature_Violation
```

Where:
- **W** = energy weight (0.0-1.0)
- **λ_E** = energy penalty coefficient (1e-5 to 1e-3)
- **λ_T** = temperature penalty coefficient (0.1-5.0)

### Comfort Ranges:
- **Winter (Oct-May)**: 20.0°C to 23.5°C
- **Summer (Jun-Sep)**: 23.0°C to 26.0°C

### Action Space:
- **Heating setpoint**: 12.0°C to 23.25°C
- **Cooling setpoint**: 23.25°C to 30.0°C

## 🚀 Next Steps

1. **Start with the demo**: Understand the concepts
2. **Install Sinergym**: When ready to run real experiments
3. **Experiment gradually**: Start with short episodes
4. **Monitor results**: Use logging and evaluation
5. **Iterate**: Try different parameters and configurations

You now have everything you need to successfully customize and train PPO agents in the 5Zone Hot environment! 🎉

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Refer to the Sinergym documentation
3. Look at the example files for reference
4. Start with the simple demo to understand concepts

Happy training! 🚀