# PPO Training in Sinergym: Complete Explanation

## Table of Contents
1. [What is PPO?](#what-is-ppo)
2. [How PPO Works](#how-ppo-works)
3. [Sinergym Environment](#sinergym-environment)
4. [Training Process](#training-process)
5. [Code Implementation](#code-implementation)
6. [Hyperparameters Explained](#hyperparameters-explained)
7. [Training Monitoring](#training-monitoring)
8. [Evaluation and Testing](#evaluation-and-testing)
9. [Common Issues and Solutions](#common-issues-and-solutions)

## What is PPO?

**PPO (Proximal Policy Optimization)** is a Deep Reinforcement Learning algorithm that learns to control systems through trial and error. In the context of Sinergym, PPO learns to control building HVAC systems to balance energy efficiency with occupant comfort.

### Key Characteristics of PPO:
- **On-Policy**: Learns from the current policy's experience
- **Proximal Updates**: Prevents too large policy changes (hence "proximal")
- **Sample Efficient**: Learns effectively from limited data
- **Stable**: Less likely to fail during training than other algorithms
- **Versatile**: Works well on many different types of problems

### Why PPO for Building Control?
1. **Continuous Actions**: Can set precise temperature setpoints (not just on/off)
2. **Multi-Objective**: Can balance energy efficiency and comfort simultaneously
3. **Adaptive**: Learns to handle different weather conditions and occupancy patterns
4. **Stable Training**: Won't suddenly "forget" what it learned

## How PPO Works

### The Learning Process

```
OBSERVATION → ACTION → REWARD → LEARNING
     ↓           ↓        ↓         ↓
Building    Heating/   Energy/   Update
State      Cooling    Comfort   Policy
```

1. **Observation**: AI observes building state (temperature, weather, energy use, etc.)
2. **Action**: AI sets heating and cooling setpoints based on current state
3. **Reward**: AI gets positive reward for good energy/comfort balance, negative for bad
4. **Learning**: AI adjusts its decision-making to get better rewards over time

### PPO Algorithm Details

#### 1. Policy Network
- Neural network that maps observations to actions
- Outputs probability distribution over actions
- Learns to predict good actions given current state

#### 2. Value Network
- Estimates expected future rewards from current state
- Helps calculate advantage (how much better an action is than expected)
- Provides learning signal for policy updates

#### 3. Proximal Updates
- Limits how much the policy can change in one update
- Prevents catastrophic forgetting
- Uses clipping to bound policy ratio changes

#### 4. Advantage Estimation
- Uses GAE (Generalized Advantage Estimation)
- Combines immediate rewards with value estimates
- Provides stable learning signal

## Sinergym Environment

### Environment Structure

```python
# Environment Creation
env = gym.make('Eplus-5zone-hot-continuous-v1')

# Observation Space (what AI can see)
obs = [
    outdoor_temperature,    # Current weather
    air_temperature,        # Indoor temperature
    air_humidity,          # Indoor humidity
    htg_setpoint,          # Current heating setpoint
    clg_setpoint,          # Current cooling setpoint
    HVAC_electricity_demand_rate,  # Energy consumption
    month, day_of_month, hour,     # Time information
    # ... more variables
]

# Action Space (what AI can control)
action = [heating_setpoint, cooling_setpoint]  # Temperature setpoints
```

### Reward Function

The reward balances energy efficiency and comfort:

```python
reward = -energy_penalty - comfort_penalty

# Energy penalty
energy_penalty = lambda_energy * HVAC_electricity_demand_rate

# Comfort penalty
if temperature < comfort_min or temperature > comfort_max:
    comfort_penalty = lambda_temperature * violation_magnitude
else:
    comfort_penalty = 0
```

### Environment Wrappers

Wrappers modify the environment to help training:

```python
# Normalize observations to similar ranges
env = NormalizeObservation(env)
# Example: temperature (20-30°C) → (0-1), energy (0-50000W) → (0-1)

# Normalize actions to [-1, 1] range
env = NormalizeAction(env)
# Example: setpoints (15-30°C) → (-1, 1)

# Add logging for analysis
env = LoggerWrapper(env)
env = CSVLogger(env)
```

## Training Process

### Phase 1: Environment Setup

```python
# Create training environment
train_env = gym.make('Eplus-5zone-hot-continuous-v1')
train_env = NormalizeObservation(train_env)
train_env = NormalizeAction(train_env)
train_env = LoggerWrapper(train_env)
train_env = CSVLogger(train_env)

# Create evaluation environment
eval_env = gym.make('Eplus-5zone-hot-continuous-v1')
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
```

### Phase 2: Model Creation

```python
model = PPO(
    policy='MlpPolicy',           # Neural network policy
    env=train_env,                # Environment to learn in
    learning_rate=0.0003,         # How fast to learn
    n_steps=2048,                 # Steps before each update
    batch_size=64,                # Training batch size
    n_epochs=10,                  # Epochs per update
    gamma=0.99,                   # Discount factor
    gae_lambda=0.95,              # Advantage estimation
    clip_range=0.2,               # PPO clipping
    ent_coef=0.01,                # Exploration coefficient
    verbose=1
)
```

### Phase 3: Training Execution

```python
# Set up evaluation callback
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="./models/",
    eval_freq=10000,              # Test every 10,000 steps
    n_eval_episodes=3,            # Episodes per evaluation
    deterministic=True             # Use best actions for testing
)

# Start training
model.learn(
    total_timesteps=200000,       # Total training steps
    callback=eval_callback,       # Use evaluation callback
    progress_bar=True             # Show progress bar
)
```

### What Happens During Training

1. **Experience Collection**: AI interacts with building for `n_steps` (2048) steps
2. **Learning Update**: AI uses this experience to improve its decision-making
3. **Evaluation**: Every 10,000 steps, test the AI on separate episodes
4. **Model Saving**: If evaluation improves, save this version as the "best model"
5. **Repeat**: Continue until `total_timesteps` (200,000) is reached

## Code Implementation

### Complete Training Script

The `direct_ppo_training.py` script provides a complete implementation with:

1. **Configuration Class**: All settings in one place
2. **Environment Creation**: Proper setup with wrappers
3. **Model Creation**: PPO with optimized hyperparameters
4. **Training Pipeline**: Complete training process
5. **Evaluation**: Comprehensive testing
6. **Comparison**: Compare with random policy

### Key Functions

#### 1. Environment Creation
```python
def create_training_environment(config):
    """Create environment with all necessary wrappers."""
    env = gym.make(config.env_name, **config.env_params)
    env = NormalizeObservation(env)
    env = NormalizeAction(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)
    return env
```

#### 2. Model Creation
```python
def create_ppo_model(env, config):
    """Create PPO model with optimized hyperparameters."""
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=1
    )
    return model
```

#### 3. Training Pipeline
```python
def train_ppo_agent(config):
    """Complete PPO training pipeline."""
    # Create environments
    train_env = create_training_environment(config)
    eval_env = create_evaluation_environment(config)
    
    # Create model
    model = create_ppo_model(train_env, config)
    
    # Set up callbacks
    callbacks = setup_training_callbacks(config, eval_env)
    
    # Train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save model
    model.save("./models/final_model")
    
    return model, train_env, eval_env
```

## Hyperparameters Explained

### Learning Rate (`learning_rate`)
- **What it does**: Controls how fast the AI learns
- **Good values**: 1e-4 to 1e-3 (0.0001 to 0.001)
- **Too high**: Unstable training, loss jumps around
- **Too low**: Very slow learning
- **Default**: 3e-4 (0.0003)

### Number of Steps (`n_steps`)
- **What it does**: How much experience to collect before each update
- **Good values**: 1024 to 4096
- **Too small**: Frequent updates, less stable
- **Too large**: Infrequent updates, slower learning
- **Default**: 2048

### Batch Size (`batch_size`)
- **What it does**: Size of data chunks used for training
- **Good values**: 32 to 128
- **Must be**: ≤ n_steps
- **Too small**: Noisy updates
- **Too large**: Memory intensive
- **Default**: 64

### Number of Epochs (`n_epochs`)
- **What it does**: How many times to reuse each batch of experience
- **Good values**: 3 to 20
- **Too few**: Underutilizes data
- **Too many**: Overfitting risk
- **Default**: 10

### Discount Factor (`gamma`)
- **What it does**: How much future rewards matter
- **Range**: 0 to 1
- **Higher**: More future-focused
- **Lower**: More immediate-focused
- **Default**: 0.99

### GAE Lambda (`gae_lambda`)
- **What it does**: Smoothing parameter for advantage estimation
- **Range**: 0 to 1
- **Higher**: More smoothing, more stable
- **Lower**: Less smoothing, more immediate
- **Default**: 0.95

### Clip Range (`clip_range`)
- **What it does**: Prevents too large policy updates
- **Good values**: 0.1 to 0.3
- **Too high**: Large updates, potential instability
- **Too low**: Very small updates, slow learning
- **Default**: 0.2

### Entropy Coefficient (`ent_coef`)
- **What it does**: Encourages exploration
- **Good values**: 0.001 to 0.1
- **Higher**: More random actions
- **Lower**: More deterministic actions
- **Default**: 0.01

## Training Monitoring

### Training Output Explanation

During training, you'll see output like:
```
---------------------------------
| rollout/            |         |
|    ep_len_mean      | 8760    |
|    ep_rew_mean      | -156.2  |
| time/               |         |
|    fps              | 45      |
|    iterations       | 100     |
|    time_elapsed     | 4563    |
|    total_timesteps  | 204800  |
| train/              |         |
|    learning_rate    | 0.0003  |
|    loss             | 0.0234  |
---------------------------------
```

**What these numbers mean:**
- **ep_len_mean**: Average episode length (8760 = full year simulation)
- **ep_rew_mean**: Average reward per episode (higher is better, less negative is better)
- **fps**: Training speed (frames/steps per second)
- **total_timesteps**: How many training steps completed so far
- **learning_rate**: Current learning rate
- **loss**: How much the AI's predictions are changing (lower = more stable)

### Callbacks for Monitoring

```python
eval_callback = EvalCallback(
    eval_env=eval_env,                    # Environment for testing
    best_model_save_path="./models/",     # Where to save best model
    log_path="./logs/",                   # Where to save logs
    eval_freq=10000,                      # Test every 10,000 steps
    n_eval_episodes=3,                    # Run 3 test episodes each time
    deterministic=True,                   # Use best actions (no randomness)
    render=False,                         # Don't show visualization
    verbose=1                             # Print evaluation results
)
```

## Evaluation and Testing

### Evaluation Process

```python
def evaluate_trained_model(model, eval_env, num_episodes=5):
    """Evaluate trained model over multiple episodes."""
    results = {
        'episode_rewards': [],
        'episode_energies': [],
        'episode_comfort_violations': []
    }
    
    for episode in range(num_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        
        terminated = truncated = False
        while not (terminated or truncated):
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Track performance metrics
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
        
        # Store episode results
        results['episode_rewards'].append(episode_reward)
        results['episode_energies'].append(episode_energy)
        results['episode_comfort_violations'].append(episode_comfort_violation)
    
    return results
```

### Comparison with Random Policy

```python
def compare_with_random_policy(eval_env, num_episodes=3):
    """Compare trained model with random actions."""
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    
    for episode in range(num_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        
        terminated = truncated = False
        while not (terminated or truncated):
            # Take random action instead of using trained model
            action = eval_env.action_space.sample()
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
        
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violation)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'mean_comfort_violation': np.mean(episode_comfort_violations)
    }
```

## Common Issues and Solutions

### Issue 1: Training is Very Slow
**Symptoms**: Training takes forever, very low FPS
**Solutions**:
```python
# Reduce episode length
config_params = {'timesteps_per_hour': 4}  # Instead of default 1

# Reduce n_steps
n_steps = 1024  # Instead of 2048

# Use CPU instead of GPU for small models
device = 'cpu'

# Reduce evaluation frequency
eval_freq = 50000  # Instead of 10000
```

### Issue 2: Training is Unstable
**Symptoms**: Training curves show big jumps, loss increases suddenly
**Solutions**:
```python
# Reduce learning rate
learning_rate = 1e-4  # Instead of 3e-4

# Reduce clip range
clip_range = 0.1  # Instead of 0.2

# Increase batch size
batch_size = 128  # Instead of 64

# Reduce epochs
n_epochs = 5  # Instead of 10
```

### Issue 3: Model Doesn't Learn
**Symptoms**: Rewards stay constant, no improvement over time
**Solutions**:
```python
# Increase learning rate
learning_rate = 1e-3  # Instead of 3e-4

# Check reward function - make sure it's not always zero
# Increase entropy coefficient for more exploration
ent_coef = 0.1  # Instead of 0.01

# Ensure environment wrappers are correct
# Check that action space is correct
```

### Issue 4: Out of Memory Errors
**Symptoms**: Computer runs out of RAM during training
**Solutions**:
```python
# Reduce batch size
batch_size = 32  # Instead of 64

# Reduce n_steps
n_steps = 1024  # Instead of 2048

# Use smaller networks
# Close other programs to free memory
```

### Issue 5: Evaluation Results Don't Match Training
**Symptoms**: Good training performance but poor evaluation
**Solutions**:
```python
# Ensure same wrappers for training and evaluation
# Use same normalization parameters
eval_env = NormalizeObservation(eval_env, 
                                mean=train_env.get_wrapper_attr('mean'),
                                var=train_env.get_wrapper_attr('var'),
                                automatic_update=False)

# Use deterministic=True for evaluation
action, _ = model.predict(obs, deterministic=True)

# Check that evaluation environment setup is identical to training
```

## Summary

PPO training in Sinergym involves:

1. **Environment Setup**: Create building simulation with proper wrappers
2. **Model Creation**: Initialize PPO with optimized hyperparameters
3. **Experience Collection**: AI interacts with building, collects experience
4. **Learning Updates**: AI uses experience to improve its policy
5. **Evaluation**: Periodic testing to monitor progress
6. **Model Saving**: Save best performing model

The key to successful PPO training is:
- **Proper hyperparameter tuning**
- **Consistent environment setup**
- **Regular evaluation and monitoring**
- **Understanding the reward function**
- **Patience during training**

The `direct_ppo_training.py` script provides a complete, self-contained implementation that you can run directly on your device without any external configuration files.