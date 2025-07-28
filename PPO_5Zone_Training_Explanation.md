# PPO Training with 5Zone Environment: Complete Explanation

## Table of Contents
1. [What is the 5Zone Environment?](#what-is-the-5zone-environment)
2. [How PPO Works with Building Control](#how-ppo-works-with-building-control)
3. [5Zone Environment Structure](#5zone-environment-structure)
4. [PPO Training Process](#ppo-training-process)
5. [Code Implementation Details](#code-implementation-details)
6. [Hyperparameters Explained](#hyperparameters-explained)
7. [Training Monitoring](#training-monitoring)
8. [Evaluation and Testing](#evaluation-and-testing)
9. [Common Issues and Solutions](#common-issues-and-solutions)

## What is the 5Zone Environment?

The **5Zone environment** is a building energy simulation that models a 5-zone commercial building with HVAC (Heating, Ventilation, and Air Conditioning) control. It's based on EnergyPlus, a building energy simulation engine.

### Key Features:
- **5 zones**: Different areas of the building with independent temperature control
- **HVAC system**: Heating and cooling equipment that consumes energy
- **Weather data**: Real weather conditions affecting building performance
- **Occupant comfort**: Temperature ranges for occupant satisfaction
- **Energy efficiency**: Minimizing energy consumption while maintaining comfort

### Building Structure:
```
5Zone Building Layout:
┌─────────────────────────────────┐
│           Zone 1                │  ← Office/Work Area
├─────────────────────────────────┤
│           Zone 2                │  ← Conference Room
├─────────────────────────────────┤
│           Zone 3                │  ← Break Room
├─────────────────────────────────┤
│           Zone 4                │  ← Storage
├─────────────────────────────────┤
│           Zone 5                │  ← Server Room
└─────────────────────────────────┘
```

## How PPO Works with Building Control

### The Control Problem:
The PPO agent must control **two continuous actions**:
1. **Heating Setpoint** (12.0°C to 23.25°C)
2. **Cooling Setpoint** (23.25°C to 30.0°C)

### The Learning Process:

#### 1. **Observation Space** (What the agent sees):
```python
# Time variables
- month, day_of_month, hour

# Environmental conditions
- outdoor_temperature, outdoor_humidity
- wind_speed, wind_direction
- diffuse_solar_radiation, direct_solar_radiation

# Building state
- htg_setpoint, clg_setpoint (current setpoints)
- air_temperature, air_humidity (zone conditions)
- people_occupant (occupancy)

# Performance metrics
- co2_emission (environmental impact)
- HVAC_electricity_demand_rate (energy consumption)
```

#### 2. **Action Space** (What the agent controls):
```python
action_space = gym.spaces.Box(
    low=np.array([12.0, 23.25], dtype=np.float32),   # [heating, cooling]
    high=np.array([23.25, 30.0], dtype=np.float32),
    shape=(2,),
    dtype=np.float32
)
```

#### 3. **Reward Function** (How performance is measured):
```python
reward_kwargs = {
    "temperature_variables": ["air_temperature"],
    "energy_variables": ["HVAC_electricity_demand_rate"],
    "range_comfort_winter": (20.0, 23.5),    # Comfort range in winter
    "range_comfort_summer": (23.0, 26.0),    # Comfort range in summer
    "energy_weight": 0.5,                     # Balance between comfort and energy
    "lambda_energy": 1.0e-4,                  # Energy penalty coefficient
    "lambda_temperature": 1.0                 # Comfort penalty coefficient
}
```

### Reward Calculation:
```
Reward = Comfort_Reward - Energy_Penalty

Where:
- Comfort_Reward = 0 if temperature is in comfort range, negative penalty otherwise
- Energy_Penalty = λ_energy × HVAC_electricity_demand_rate
```

## 5Zone Environment Structure

### Environment Configuration:
```python
env_kwargs = {
    # Building and weather
    "building_file": "5ZoneAutoDXVAV.epJSON",
    "weather_files": "USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw",
    
    # Time variables (temporal context)
    "time_variables": ["month", "day_of_month", "hour"],
    
    # Observation variables (building state)
    "variables": {
        "outdoor_temperature": ("Site Outdoor Air DryBulb Temperature", "Environment"),
        "air_temperature": ("Zone Air Temperature", "SPACE5-1"),
        "HVAC_electricity_demand_rate": ("Facility Total HVAC Electricity Demand Rate", "Whole Building"),
        # ... more variables
    },
    
    # Action variables (control points)
    "actuators": {
        "Heating_Setpoint_RL": ("Schedule:Compact", "Schedule Value", "HTG-SETP-SCH"),
        "Cooling_Setpoint_RL": ("Schedule:Compact", "Schedule Value", "CLG-SETP-SCH")
    },
    
    # Action space (continuous control)
    "action_space": gym.spaces.Box(
        low=np.array([12.0, 23.25], dtype=np.float32),
        high=np.array([23.25, 30.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32
    )
}
```

### Weather Files Available:
- **Hot weather**: `USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw`
- **Mixed weather**: `USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw`
- **Cool weather**: `USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw`

## PPO Training Process

### 1. **Environment Creation**:
```python
def create_5zone_environment(weather_file, config_params, is_eval=False):
    # Define all environment parameters directly in code
    env_kwargs = {
        "building_file": "5ZoneAutoDXVAV.epJSON",
        "weather_files": weather_file,
        "variables": {...},  # All observation variables
        "actuators": {...},  # All action variables
        "action_space": gym.spaces.Box(...),
        "reward": LinearReward,
        "reward_kwargs": {...}
    }
    
    env = EplusEnv(**env_kwargs)
    env = Monitor(env)  # For logging
    return env
```

### 2. **PPO Model Creation**:
```python
def create_ppo_model(env, config):
    model = PPO(
        "MlpPolicy",                    # Multi-layer perceptron policy
        env,
        learning_rate=3e-4,            # How fast to learn
        n_steps=2048,                  # Steps per update
        batch_size=64,                 # Batch size for training
        n_epochs=10,                   # Epochs per update
        gamma=0.99,                    # Discount factor
        gae_lambda=0.95,              # GAE lambda
        clip_range=0.2,                # PPO clip range
        ent_coef=0.01,                # Entropy coefficient
        policy_kwargs={
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],  # Network architecture
            "activation_fn": "tanh"
        }
    )
    return model
```

### 3. **Training Process**:
```python
def train_ppo_agent(config, train_env, eval_env):
    # Create PPO model
    model = create_ppo_model(train_env, config)
    
    # Setup callbacks for monitoring
    callbacks = setup_training_callbacks(config, eval_env)
    
    # Train the model
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    return model
```

## Code Implementation Details

### Configuration Class:
```python
class PPOTrainingConfig:
    def __init__(self):
        # Environment settings
        self.building_file = "5ZoneAutoDXVAV.epJSON"
        self.weather_file = "USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw"
        self.experiment_name = f"ppo_5zone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Training parameters
        self.total_timesteps = 100000
        self.eval_freq = 5000
        self.save_freq = 10000
        self.n_eval_episodes = 5
        
        # PPO hyperparameters
        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01
        
        # Environment parameters
        self.config_params = {
            "runperiod": (1, 1, 1991, 1, 31, 1991),  # January 1991
            "timesteps_per_hour": 1
        }
```

### Training Callbacks:
```python
def setup_training_callbacks(config, eval_env):
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{config.experiment_name}/",
        log_path=f"./logs/{config.experiment_name}/",
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=f"./checkpoints/{config.experiment_name}/",
        name_prefix="ppo_model"
    )
    callbacks.append(checkpoint_callback)
    
    return callbacks
```

## Hyperparameters Explained

### PPO Hyperparameters:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `learning_rate` | 3e-4 | How fast the agent learns from experiences |
| `n_steps` | 2048 | Number of steps to collect before updating policy |
| `batch_size` | 64 | Number of samples per training batch |
| `n_epochs` | 10 | Number of training epochs per update |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | GAE (Generalized Advantage Estimation) parameter |
| `clip_range` | 0.2 | PPO clip range to prevent large policy updates |
| `ent_coef` | 0.01 | Entropy coefficient for exploration |

### Environment Hyperparameters:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `total_timesteps` | 100000 | Total training steps |
| `eval_freq` | 5000 | Evaluate every N steps |
| `save_freq` | 10000 | Save model every N steps |
| `n_eval_episodes` | 5 | Episodes for evaluation |

## Training Monitoring

### What to Monitor:

1. **Training Loss**: Policy and value function losses
2. **Episode Rewards**: Average reward per episode
3. **Energy Consumption**: HVAC electricity demand
4. **Comfort Violations**: Temperature outside comfort range
5. **Learning Progress**: Improvement over time

### Monitoring Tools:

```python
# TensorBoard logging
tensorboard_log=f"./tensorboard_logs/{config.experiment_name}/"

# Evaluation metrics
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/{config.experiment_name}/",
    log_path=f"./logs/{config.experiment_name}/",
    eval_freq=config.eval_freq,
    n_eval_episodes=config.n_eval_episodes
)
```

## Evaluation and Testing

### Evaluation Metrics:

```python
def evaluate_trained_model(model, eval_env, n_episodes=10):
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_energy": np.mean(episode_energy_consumption),
        "mean_comfort_violations": np.mean(episode_comfort_violations)
    }
    return metrics
```

### Key Performance Indicators:

1. **Reward**: Higher is better (comfort - energy penalty)
2. **Energy Consumption**: Lower is better
3. **Comfort Violations**: Lower is better
4. **Episode Length**: Should be consistent

## Common Issues and Solutions

### 1. **Environment Creation Issues**:
```python
# Problem: Missing EnergyPlus
# Solution: Install EnergyPlus and pyenergyplus
pip install epluspy

# Problem: Missing weather files
# Solution: Ensure weather files are in the correct directory
```

### 2. **Training Issues**:
```python
# Problem: Training is too slow
# Solution: Reduce n_steps or increase learning_rate
config.n_steps = 1024  # Smaller batches
config.learning_rate = 5e-4  # Faster learning

# Problem: Poor convergence
# Solution: Adjust hyperparameters
config.ent_coef = 0.05  # More exploration
config.clip_range = 0.1  # Smaller policy updates
```

### 3. **Memory Issues**:
```python
# Problem: Out of memory
# Solution: Reduce batch size and n_steps
config.batch_size = 32
config.n_steps = 1024
```

## Running the Training

### Quick Start:
```bash
# Run the complete training pipeline
python ppo_5zone_training.py

# Run tests first
python test_ppo_5zone.py
```

### Expected Output:
```
🏢 5Zone PPO Training Pipeline
==================================================

🔧 Creating environments...
✅ 5Zone Environment created successfully
   - Building: 5ZoneAutoDXVAV.epJSON
   - Weather: USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw
   - Observation space: Box(13,)
   - Action space: Box(2,)
   - Action range: Heating [12.0, 23.2°C]
   - Action range: Cooling [23.2, 30.0°C]

🎯 Training PPO agent...
🤖 Creating PPO model with configuration:
   - Learning rate: 0.0003
   - N steps: 2048
   - Batch size: 64
   - N epochs: 10
   - Gamma: 0.99
   - Clip range: 0.2
   - Entropy coefficient: 0.01

🎯 Training started...
[Training progress...]

📊 Evaluating trained model...
📈 Evaluation Results:
   - Mean reward: 245.67 ± 12.34
   - Mean episode length: 744.0
   - Mean energy consumption: 1234.56
   - Mean comfort violations: 2.3

✅ Training completed successfully!
```

This implementation provides a complete, self-contained PPO training solution for the 5Zone environment without requiring any external configuration files.