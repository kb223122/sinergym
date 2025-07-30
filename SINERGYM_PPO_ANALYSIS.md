# Sinergym PPO Training Analysis and Setup Guide

## Overview

Sinergym is a building simulation environment for reinforcement learning that allows AI agents to learn how to control HVAC (Heating, Ventilation, and Air Conditioning) systems in buildings. The repository contains several PPO (Proximal Policy Optimization) training implementations designed for different skill levels.

## Repository Structure Analysis

### Key Files for PPO Training

1. **`ppo_beginner_tutorial.py`** - Complete beginner-friendly tutorial
2. **`ppo_training_guide_detailed.py`** - Advanced training guide with detailed explanations
3. **`train_agent_simple.py`** - Simple training script
4. **`PPO_Training_Guide.md`** - Documentation guide

### Environment Structure

The Sinergym environments are built on top of EnergyPlus building simulation software and provide:

- **Observation Space**: Building sensor data (temperatures, humidity, energy consumption, etc.)
- **Action Space**: HVAC control actions (heating/cooling setpoints)
- **Reward Function**: Balances energy efficiency and occupant comfort

## PPO Algorithm Explanation

### What is PPO?

PPO (Proximal Policy Optimization) is a state-of-the-art reinforcement learning algorithm that:

1. **Learns through trial and error** - The agent tries different actions and learns from the rewards
2. **Balances exploration and exploitation** - Tries new strategies while using what it has learned
3. **Prevents catastrophic updates** - Uses a "clipped" objective to avoid large policy changes
4. **Works well with continuous actions** - Perfect for HVAC control where setpoints are continuous values

### How PPO Works in Building Control

```
Building State (Temperature, Weather, etc.)
    ↓
PPO Agent (Neural Network)
    ↓
Action (Heating/Cooling Setpoints)
    ↓
Building Simulation (EnergyPlus)
    ↓
Reward (Energy Efficiency + Comfort)
    ↓
Learning Update
```

## Code Analysis

### 1. Beginner Tutorial (`ppo_beginner_tutorial.py`)

**Key Features:**
- Step-by-step explanations for every line of code
- Simple configuration with good default values
- Comprehensive error handling and user guidance
- Built-in evaluation and comparison with random actions

**Main Components:**

```python
# Environment Creation
def create_simple_environment(env_name, experiment_name, for_training=True):
    # Creates environment with helpful wrappers:
    # - NormalizeObservation: Scales sensor readings
    # - NormalizeAction: Scales actions to [-1, 1]
    # - LoggerWrapper: Records interactions
    # - CSVLogger: Saves data to files

# PPO Model Creation
def create_simple_ppo_model(env):
    # Creates PPO with good default hyperparameters:
    # - learning_rate=0.0003
    # - n_steps=2048
    # - batch_size=64

# Training Process
def train_simple_ppo():
    # 1. Creates training and evaluation environments
    # 2. Sets up evaluation callback
    # 3. Trains the model
    # 4. Saves the best model
```

### 2. Detailed Training Guide (`ppo_training_guide_detailed.py`)

**Key Features:**
- Advanced configuration management
- Comprehensive hyperparameter tuning
- Detailed performance analysis and plotting
- Professional-grade training pipeline

**Main Components:**

```python
class PPOConfig:
    # Centralized configuration management
    # All hyperparameters in one place
    # Easy to modify and experiment

def create_training_environment(config):
    # Advanced environment setup
    # Proper normalization
    # Custom reward functions

def train_ppo_agent(config):
    # Professional training pipeline
    # Multiple callbacks
    # Comprehensive logging
```

## Environment Types Available

Sinergym provides multiple environment variants:

### Building Types
- **5Zone**: 5-zone office building (most common for tutorials)
- **Office**: Office building
- **Datacenter**: Data center building
- **Warehouse**: Warehouse building
- **Shop**: Retail building

### Climate Conditions
- **Hot**: Hot climate conditions
- **Mixed**: Mixed climate conditions  
- **Cool**: Cool climate conditions

### Action Spaces
- **Continuous**: Continuous action space (recommended for PPO)
- **Discrete**: Discrete action space

### Stochasticity
- **Stochastic**: Includes weather uncertainty
- **Deterministic**: Fixed weather conditions

**Example Environment Names:**
- `Eplus-5zone-hot-continuous-v1` (recommended for beginners)
- `Eplus-office-mixed-discrete-stochastic-v1`
- `Eplus-datacenter-cool-continuous-v1`

## Setup Instructions

### Prerequisites

1. **Python Environment**: Python 3.10+ recommended
2. **EnergyPlus**: Building simulation software (version 24.1.0)
3. **Dependencies**: See requirements below

### Installation Steps

#### Option 1: Docker (Recommended)

```bash
# Build the Docker image with all dependencies
docker build -t sinergym:latest --build-arg SINERGYM_EXTRAS="drl" .

# Run the container
docker run -it --rm sinergym:latest

# Inside container, run training
python ppo_beginner_tutorial.py
```

#### Option 2: Manual Installation

```bash
# 1. Create virtual environment
python3 -m venv sinergym_env
source sinergym_env/bin/activate

# 2. Install Sinergym
pip install -e .

# 3. Install EnergyPlus (required for building simulation)
# Download from: https://energyplus.net/downloads
# Extract to /usr/local/EnergyPlus-24.1.0
export ENERGYPLUS_INSTALLATION_DIR=/usr/local/EnergyPlus-24.1.0

# 4. Install additional dependencies
pip install stable-baselines3 matplotlib wandb
```

### Required Dependencies

```python
# Core dependencies
gymnasium>=1.0.0
numpy>=2.2.0
pandas>=2.2.2
pyyaml>=6.0.2

# Building simulation
eppy>=0.5.63
epw>=1.2.dev2

# Reinforcement learning
stable-baselines3>=2.0.0

# Optional (for advanced features)
matplotlib  # For plotting
wandb       # For experiment tracking
jupyter     # For notebooks
```

## Training Configuration

### Beginner Configuration

```python
# Simple configuration for beginners
ENV_NAME = 'Eplus-5zone-hot-continuous-v1'
TOTAL_TIMESTEPS = 50000
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 64
```

### Advanced Configuration

```python
# Advanced configuration for better performance
class PPOConfig:
    def __init__(self):
        self.total_timesteps = 1000000  # 1M steps for better learning
        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
```

## Training Process

### 1. Environment Setup

```python
# Create environment with wrappers
env = gym.make('Eplus-5zone-hot-continuous-v1')
env = NormalizeObservation(env)  # Scale observations
env = NormalizeAction(env)       # Scale actions
env = LoggerWrapper(env)         # Log interactions
```

### 2. Model Creation

```python
# Create PPO model
model = PPO(
    policy='MlpPolicy',         # Neural network policy
    env=env,                    # Environment
    learning_rate=0.0003,       # Learning rate
    n_steps=2048,              # Steps per update
    batch_size=64,             # Batch size
    verbose=1                  # Print progress
)
```

### 3. Training

```python
# Set up evaluation callback
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,
    n_eval_episodes=3
)

# Train the model
model.learn(
    total_timesteps=50000,
    callback=eval_callback,
    progress_bar=True
)
```

### 4. Evaluation

```python
# Test the trained model
results = evaluate_model(model, num_episodes=5)
print(f"Average reward: {results['mean_reward']:.2f}")
print(f"Energy efficiency: {results['mean_energy']:.2f} kWh")
```

## Key Metrics

### Performance Metrics

1. **Reward**: Overall performance (higher is better)
2. **Energy Consumption**: Total energy used (lower is better)
3. **Comfort Violations**: Temperature deviations from comfort range (lower is better)
4. **Episode Length**: Number of timesteps per episode

### Training Metrics

1. **Learning Rate**: How fast the agent learns
2. **Policy Loss**: How much the policy is changing
3. **Value Loss**: How well the agent predicts future rewards
4. **Entropy**: How much the agent explores vs exploits

## Common Issues and Solutions

### 1. EnergyPlus Not Found

**Problem**: `ModuleNotFoundError: No module named 'pyenergyplus'`

**Solution**: Install EnergyPlus and set environment variables
```bash
export ENERGYPLUS_INSTALLATION_DIR=/path/to/EnergyPlus
export PYTHONPATH=$PYTHONPATH:$ENERGYPLUS_INSTALLATION_DIR
```

### 2. Environment Not Found

**Problem**: `Environment doesn't exist`

**Solution**: Check environment name case sensitivity
```python
# Correct
env = gym.make('Eplus-5zone-hot-continuous-v1')

# Wrong
env = gym.make('Eplus-5Zone-hot-continuous-v1')  # Wrong case
```

### 3. Training Too Slow

**Solution**: Reduce complexity
```python
TOTAL_TIMESTEPS = 10000  # Start smaller
N_STEPS = 1024          # Reduce batch size
BATCH_SIZE = 32         # Smaller batches
```

## Advanced Features

### 1. Custom Reward Functions

```python
# Custom reward function
def custom_reward(obs, action, reward, info):
    energy_penalty = info.get('total_power_demand', 0) * 0.1
    comfort_reward = -info.get('total_temperature_violation', 0)
    return reward + comfort_reward - energy_penalty
```

### 2. Multi-Environment Training

```python
# Train on multiple environments
envs = [
    'Eplus-5zone-hot-continuous-v1',
    'Eplus-5zone-mixed-continuous-v1',
    'Eplus-5zone-cool-continuous-v1'
]
```

### 3. Hyperparameter Tuning

```python
# Grid search for best hyperparameters
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        model = PPO(learning_rate=lr, batch_size=bs, ...)
        # Train and evaluate
```

## Best Practices

### 1. Start Simple
- Use the beginner tutorial first
- Start with small training runs (10K-50K steps)
- Use the 5Zone environment for learning

### 2. Monitor Training
- Watch the reward curves
- Check for overfitting (training reward increases but eval reward decreases)
- Use evaluation callbacks

### 3. Experiment Gradually
- Change one hyperparameter at a time
- Keep a log of experiments
- Use version control for code changes

### 4. Validate Results
- Always compare with random actions
- Test on multiple episodes
- Check for reasonable energy consumption

## Next Steps

1. **Run the beginner tutorial** to understand the basics
2. **Experiment with different environments** to see how they behave
3. **Try different hyperparameters** to improve performance
4. **Implement custom reward functions** for specific objectives
5. **Use advanced features** like multi-environment training
6. **Deploy in real buildings** (with proper safety measures)

## Resources

- **Documentation**: https://ugr-sail.github.io/sinergym/
- **GitHub**: https://github.com/ugr-sail/sinergym
- **EnergyPlus**: https://energyplus.net/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

This analysis provides a comprehensive understanding of the Sinergym PPO training code and how to set it up for building control applications.