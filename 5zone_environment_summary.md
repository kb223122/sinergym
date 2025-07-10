# 5Zone Hot Continuous Environment - Complete Reference

## Environment Overview

**Environment ID:** `Eplus-5zone-hot-continuous-v1`

- **Building Model:** 5ZoneAutoDXVAV.epJSON (5-zone commercial building)
- **HVAC System:** Variable Air Volume (VAV) with auto-sizing
- **Climate:** Hot desert climate (Davis-Monthan AFB, Arizona)
- **Control Zone:** SPACE5-1 (central zone)
- **Building Area:** ~500 m²

## Observation Space (17 variables)

| # | Variable | Unit | Range | Description |
|---|----------|------|-------|-------------|
| **Time Variables (3)** |
| 1 | month | int | 1-12 | Current simulation month |
| 2 | day_of_month | int | 1-31 | Current day of the month |
| 3 | hour | int | 1-24 | Current hour of the day |
| **Environmental & Zone Variables (13)** |
| 4 | outdoor_temperature | °C | -50 to 60 | Site outdoor air dry bulb temperature |
| 5 | outdoor_humidity | % | 0 to 100 | Site outdoor air relative humidity |
| 6 | wind_speed | m/s | 0 to 40 | Site wind speed |
| 7 | wind_direction | degrees | 0 to 360 | Site wind direction |
| 8 | diffuse_solar_radiation | W/m² | 0 to 1000 | Site diffuse solar radiation rate |
| 9 | direct_solar_radiation | W/m² | 0 to 1000 | Site direct solar radiation rate |
| 10 | htg_setpoint | °C | 12 to 30 | Zone heating setpoint (SPACE5-1) |
| 11 | clg_setpoint | °C | 18 to 35 | Zone cooling setpoint (SPACE5-1) |
| 12 | air_temperature | °C | 10 to 40 | Zone air temperature (SPACE5-1) |
| 13 | air_humidity | % | 0 to 100 | Zone air relative humidity (SPACE5-1) |
| 14 | people_occupant | people | 0 to 50 | Zone occupant count (SPACE5-1) |
| 15 | co2_emission | kg | 0 to 1000 | Total CO2 emissions |
| 16 | HVAC_electricity_demand_rate | W | 0 to 50000 | HVAC electricity demand rate |
| **Energy Meter Variables (1)** |
| 17 | total_electricity_HVAC | J | 0 to 1e8 | Total HVAC electricity consumption |

### Observation Space Properties
- **Type:** `gymnasium.spaces.Box`
- **Shape:** `(17,)`
- **Data Type:** `float32`
- **Bounds:** `[-5e7, 5e7]`

## Action Space (2 variables)

| # | Variable | Unit | Range | Description |
|---|----------|------|-------|-------------|
| 1 | Heating_Setpoint_RL | °C | 12.0 - 23.25 | Heating setpoint temperature control |
| 2 | Cooling_Setpoint_RL | °C | 23.25 - 30.0 | Cooling setpoint temperature control |

### Action Space Properties
- **Type:** `gymnasium.spaces.Box`
- **Shape:** `(2,)`
- **Data Type:** `float32`
- **Constraints:** 
  - Heating setpoint ≤ Cooling setpoint
  - Minimum deadband: 0.25°C

### Sample Control Strategies

| Strategy | Action [H, C] | Description |
|----------|---------------|-------------|
| Energy Saving (Summer) | [16.0, 28.0] | Wide deadband, high cooling setpoint |
| Comfort Priority (Summer) | [20.0, 24.0] | Narrow deadband, optimal comfort |
| Energy Saving (Winter) | [18.0, 26.0] | Low heating, moderate cooling |
| Comfort Priority (Winter) | [21.0, 23.5] | Warm heating, close deadband |
| Extreme Energy Saving | [12.0, 30.0] | Maximum allowable deadband |

## Reward Function

### Default: LinearReward

**Formula:** `R = -W × λE × power - (1-W) × λT × temperature_violation`

**Default Parameters:**
```json
{
  "temperature_variables": ["air_temperature"],
  "energy_variables": ["HVAC_electricity_demand_rate"],
  "range_comfort_winter": [20.0, 23.5],
  "range_comfort_summer": [23.0, 26.0],
  "summer_start": [6, 1],
  "summer_final": [9, 30],
  "energy_weight": 0.5,
  "lambda_energy": 1e-4,
  "lambda_temperature": 1.0
}
```

### Available Reward Classes

| Class | Purpose | Description |
|-------|---------|-------------|
| LinearReward | Standard linear combination | Balances energy and comfort |
| EnergyCostLinearReward | Includes energy cost | Adds economic considerations |
| ExpReward | Exponential penalty | Heavily penalizes comfort violations |
| HourlyLinearReward | Time-dependent weighting | Different weights for work/off hours |
| NormalizedLinearReward | Normalized rewards | Bounded values for stable training |
| MultiZoneReward | Multi-zone environments | Individual comfort ranges per zone |

### Custom Reward Examples

**Energy-Focused (80% energy weight):**
```json
{
  "energy_weight": 0.8,
  "lambda_energy": 1e-4,
  "lambda_temperature": 0.5,
  "range_comfort_winter": [18.0, 25.0],
  "range_comfort_summer": [22.0, 28.0]
}
```

**Comfort-Focused (20% energy weight):**
```json
{
  "energy_weight": 0.2,
  "lambda_energy": 1e-4,
  "lambda_temperature": 2.0,
  "range_comfort_winter": [20.5, 22.5],
  "range_comfort_summer": [23.5, 25.5]
}
```

## Episode Structure

### Temporal Configuration
- **Episode Length:** 8,760 timesteps (1 full year)
- **Timestep Size:** 1 hour
- **Simulation Period:** January 1 - December 31, 1991
- **Steps per Day:** 24

### Monthly Breakdown
| Month | Days | Steps | Cumulative |
|-------|------|-------|------------|
| Jan | 31 | 744 | 744 |
| Feb | 28 | 672 | 1,416 |
| Mar | 31 | 744 | 2,160 |
| Apr | 30 | 720 | 2,880 |
| May | 31 | 744 | 3,624 |
| Jun | 30 | 720 | 4,344 |
| Jul | 31 | 744 | 5,088 |
| Aug | 31 | 744 | 5,832 |
| Sep | 30 | 720 | 6,552 |
| Oct | 31 | 744 | 7,296 |
| Nov | 30 | 720 | 8,016 |
| Dec | 31 | 744 | 8,760 |

### Seasonal Characteristics (Arizona Climate)
- **Winter (Dec-Feb):** Mild temperatures, minimal heating load
- **Spring (Mar-May):** Moderate temperatures, transitional period  
- **Summer (Jun-Sep):** Very hot, high cooling load, comfort critical
- **Fall (Oct-Nov):** Cooling temperatures, reduced HVAC load

## Workflow

### Basic Usage
```python
import gymnasium as gym
import sinergym

# Create environment
env = gym.make('Eplus-5zone-hot-continuous-v1')

# Reset for new episode
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # (17,)

# Episode loop
total_reward = 0
for step in range(env.timestep_per_episode):  # 8760 steps
    # Select action (heating and cooling setpoints)
    action = env.action_space.sample()  # Random policy
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Monitor performance
    energy = info['total_power_demand']
    comfort = info['total_temperature_violation']
    
    if terminated or truncated:
        break

print(f"Episode reward: {total_reward:.2f}")
env.close()
```

### Reinforcement Learning Training
```python
import stable_baselines3 as sb3

env = gym.make('Eplus-5zone-hot-continuous-v1')
model = sb3.PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)  # ~11 episodes
model.save("ppo_5zone_hvac")
```

## Performance Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| Total Episode Reward | Sum of all timestep rewards | Higher is better |
| Energy Consumption | Total HVAC electricity (kWh) | Lower is better |
| Comfort Violations | Temperature deviations (°C⋅hours) | Lower is better |
| Peak Demand | Maximum power demand (kW) | Lower is better |
| Comfort Hours | Hours within comfort range (%) | Higher is better |
| Energy Efficiency | Comfort per unit energy | Higher is better |

## Performance Benchmarks

| Controller Type | Typical Reward Range | Description |
|----------------|---------------------|-------------|
| Random Policy | -500 to -1000 | Baseline comparison |
| Rule-Based Controller | -200 to -400 | Simple thermostat control |
| Well-Tuned RL Agent | -100 to -200 | Optimized reinforcement learning |
| Expert Human Operator | -150 to -250 | Skilled building operator |

## Environment Variants

| Environment ID | Description |
|----------------|-------------|
| `Eplus-5zone-hot-continuous-v1` | Standard continuous control |
| `Eplus-5zone-hot-discrete-v1` | Discrete action space (10 actions) |
| `Eplus-5zone-hot-continuous-stochastic-v1` | Continuous with weather variability |
| `Eplus-5zone-hot-discrete-stochastic-v1` | Discrete with weather variability |

## Advanced Features

### Weather Variability (Stochastic Environments)
```python
weather_variability = {
    'Dry Bulb Temperature': (1.0, 0.0, 24.0),  # (sigma, mu, tau)
    'Relative Humidity': (5.0, 0.0, 12.0),
    'Wind Speed': (0.5, 0.0, 6.0)
}
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
```

### Environment Wrappers
- `LoggerWrapper`: Log interactions to CSV files
- `MultiObjectiveReward`: Handle multiple objectives
- `NormalizeObservation`: Normalize observations to [0,1]
- `DiscretizeEnv`: Convert to discrete action space
- `PreviousObservationWrapper`: Include history
- `DatetimeWrapper`: Add datetime features

## Optimization Recommendations

### Training Strategies
1. **Reward Shaping:** Start with balanced energy_weight (0.5)
2. **Curriculum Learning:** Begin with relaxed comfort ranges
3. **Action Scaling:** Normalize to [-1, 1] for better convergence
4. **Observation Normalization:** Improve training stability
5. **Early Stopping:** Monitor validation performance
6. **Hyperparameter Tuning:** Optimize learning parameters

### Common Issues & Solutions

**High Energy Consumption:**
- Increase energy_weight in reward function
- Penalize extreme setpoint values
- Add equipment efficiency considerations

**Poor Comfort Performance:**
- Decrease energy_weight in reward function
- Tighten comfort temperature ranges
- Increase lambda_temperature scaling

**Unstable Training:**
- Normalize observations and actions
- Reduce learning rate
- Use reward clipping

## Research Applications

1. Multi-objective HVAC optimization
2. Demand response and grid integration
3. Occupancy-based control strategies
4. Predictive maintenance scheduling
5. Energy-comfort trade-off analysis
6. Building automation system design
7. Climate adaptation strategies
8. Renewable energy integration

## Summary

The `Eplus-5zone-hot-continuous-v1` environment provides:
- ✅ Realistic 5-zone building simulation with VAV HVAC system
- ✅ Comprehensive 17-variable observation space
- ✅ Continuous 2-variable action space for setpoint control
- ✅ Customizable reward functions balancing energy and comfort
- ✅ Full annual simulation (8,760 timesteps)
- ✅ Hot desert climate with challenging cooling loads
- ✅ Suitable for reinforcement learning research and HVAC optimization

This environment is ideal for developing and testing intelligent building control systems using reinforcement learning techniques.