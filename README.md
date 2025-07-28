# 5Zone Environment PPO Training with Custom Reward Weights and Run Periods

This repository contains complete PPO training examples for the Sinergym `Eplus-5zone-hot-continuous-v1` environment with customizable reward weights and run periods.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install sinergym[drl] stable-baselines3 gymnasium numpy pandas matplotlib
```

### 2. Run Simple Training
```bash
python simple_ppo_training.py
```

### 3. View Available Configurations
```bash
python environment_config.py
```

## 📁 Files Overview

### Core Training Scripts
- **`simple_ppo_training.py`** - Easy-to-use PPO training with configuration utility
- **`ppo_5zone_training.py`** - Comprehensive PPO training with detailed setup
- **`environment_config.py`** - Configuration utility for reward weights and run periods

### Testing and Demo Scripts
- **`quick_reward_demo.py`** - Quick demonstration of reward weight effects
- **`5zone_customization_guide.py`** - Comprehensive guide with practical examples
- **`simple_reward_test.py`** - Detailed reward function testing

### Documentation
- **`IMPLEMENTATION_GUIDE.md`** - Complete implementation reference
- **`README.md`** - This file

## 🎯 Key Features

### Customizable Reward Weights
- **`energy_weight`** (0.0-1.0): Controls balance between energy and comfort
- **`lambda_energy`**: Scales energy consumption penalty
- **`lambda_temperature`**: Scales comfort violation penalty

### Configurable Run Periods
- **`timestep_per_hour`**: Controls simulation granularity (1, 2, 4 hours)
- **`runperiod`**: Controls episode duration (seasonal periods)

## 🔧 Configuration Options

### Reward Configurations
```python
# Available reward configurations:
'default'           # Balanced energy and comfort
'energy_focused'    # Prioritizes energy savings
'comfort_focused'   # Prioritizes occupant comfort
'extreme_energy'    # Maximum energy savings
'extreme_comfort'   # Maximum comfort priority
'custom'           # Custom configuration
```

### Run Period Configurations
```python
# Available run period configurations:
'default'              # Full year with 1-hour timesteps
'summer_only'          # Summer months (June-August)
'winter_only'          # Winter months (December-February)
'spring_only'          # Spring months (March-May)
'two_hour_timesteps'   # Full year with 2-hour timesteps
'four_hour_timesteps'  # Full year with 4-hour timesteps
'summer_two_hour'      # Summer with 2-hour timesteps
```

### Training Configurations
```python
# Available training configurations:
'default'        # Standard training (50k timesteps)
'quick_test'     # Quick test training (10k timesteps)
'long_training'  # Extended training (200k timesteps)
'high_lr'        # High learning rate
'low_lr'         # Low learning rate
```

## 🚀 Usage Examples

### Example 1: Energy-Focused Training (Summer Only)
```python
# In simple_ppo_training.py, modify these lines:
reward_config_name = 'energy_focused'
run_period_config_name = 'summer_only'
training_config_name = 'quick_test'
```

### Example 2: Comfort-Focused Training (Full Year)
```python
# In simple_ppo_training.py, modify these lines:
reward_config_name = 'comfort_focused'
run_period_config_name = 'default'
training_config_name = 'default'
```

### Example 3: Custom Configuration
```python
# In simple_ppo_training.py, modify these lines:
reward_config_name = 'custom'
run_period_config_name = 'two_hour_timesteps'
training_config_name = 'long_training'
```

## 📊 Reward Weight Effects

| Configuration | Energy Weight | Lambda Energy | Lambda Temp | Effect |
|---------------|---------------|---------------|-------------|---------|
| Default | 0.50 | 1.00e-04 | 1.00 | Balanced |
| Energy-Focused | 0.80 | 2.00e-04 | 0.50 | Prioritizes energy |
| Comfort-Focused | 0.20 | 5.00e-05 | 2.00 | Prioritizes comfort |
| Extreme Energy | 0.90 | 5.00e-04 | 0.10 | Maximum energy savings |
| Extreme Comfort | 0.10 | 1.00e-05 | 5.00 | Maximum comfort priority |

## ⏱️ Run Period Effects

| Configuration | Timesteps/Hour | Duration | Episodes | Use Case |
|---------------|----------------|----------|----------|----------|
| Default | 1 | Full year | 8760 | Comprehensive analysis |
| Summer Only | 1 | 3 months | 2208 | Summer optimization |
| Winter Only | 1 | 3 months | 2160 | Winter optimization |
| 2-Hour | 2 | Full year | 4380 | Faster training |
| 4-Hour | 4 | Full year | 2190 | Very fast training |

## 🎯 Training Process

### 1. Environment Setup
- Creates custom reward function with specified weights
- Configures run period with specified timesteps and duration
- Verifies all configurations are applied correctly

### 2. PPO Training
- Uses Stable-Baselines3 PPO implementation
- Includes evaluation callbacks for monitoring
- Saves checkpoints and best model

### 3. Evaluation
- Evaluates trained model on multiple episodes
- Reports mean reward and performance metrics
- Saves final model with timestamp

## 📁 Output Files

After training, you'll find:
- **`./models/`** - Saved model checkpoints and final model
- **`./logs/`** - Training logs and evaluation results
- **Console output** - Real-time training progress and final results

## 🔍 Verification Methods

### 1. Environment Configuration Verification
```python
# Verify reward function parameters
print(f"Energy Weight: {env.reward_fn.energy_weight}")
print(f"Lambda Energy: {env.reward_fn.lambda_energy}")
print(f"Lambda Temperature: {env.reward_fn.lambda_temperature}")

# Verify run period settings
print(f"Timestep per Hour: {env.timestep_per_hour}")
print(f"Run Period: {env.runperiod}")
print(f"Timesteps per Episode: {env.timestep_per_episode}")
```

### 2. Reward Calculation Testing
```python
# Test reward calculation
test_obs = {
    'air_temperature': 28.0,
    'HVAC_electricity_demand_rate': 8000.0,
    'month': 7, 'day_of_month': 15, 'hour': 14
}
reward, terms = env.reward_fn(test_obs)
print(f"Reward: {reward}")
```

## 🛠️ Customization Guide

### Modifying Reward Weights
```python
# In simple_ppo_training.py, change these lines:
reward_config_name = 'energy_focused'  # or any other configuration
```

### Modifying Run Periods
```python
# In simple_ppo_training.py, change these lines:
run_period_config_name = 'summer_only'  # or any other configuration
```

### Adding Custom Configurations
```python
# In environment_config.py, add to the configs dictionaries:
'my_custom_reward': {
    'energy_weight': 0.6,
    'lambda_energy': 1.5e-4,
    'lambda_temperature': 1.2,
    'description': 'My custom configuration'
}
```

## 🎯 Best Practices

1. **Start with balanced settings**: Use `energy_weight=0.5` initially
2. **Test different configurations**: Experiment with various lambda values
3. **Monitor both metrics**: Track both energy consumption and comfort violations
4. **Verify configurations**: Always check that your changes are applied correctly
5. **Consider your use case**: Choose timesteps and periods based on your research goals

## 🚨 Troubleshooting

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

## 📚 Additional Resources

- **Sinergym Documentation**: https://ugr-sail.github.io/sinergym/
- **Stable-Baselines3 Documentation**: https://stable-baselines3.readthedocs.io/
- **5Zone Environment Guide**: See `IMPLEMENTATION_GUIDE.md`

## 🎉 Example Results

After running the training, you should see output like:
```
✅ TRAINING COMPLETED SUCCESSFULLY!
==================================================
Configuration: energy_focused + summer_only + quick_test
Final Mean Reward: -1.2345
Model saved to: ./models/ppo_5zone_final_20241201_143022
```

This complete setup allows you to easily customize and train PPO agents on the 5Zone environment with your specific reward weights and run periods!
