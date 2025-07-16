# PPO Training and Evaluation Guide for Sinergym 5Zone Environment

## Table of Contents
1. [Introduction to Deep RL and PPO](#introduction)
2. [Understanding the Code Structure](#code-structure)
3. [Step-by-Step Training Process](#training-process)
4. [Step-by-Step Evaluation Process](#evaluation-process)
5. [Code Examples with Line-by-Line Explanations](#code-examples)
6. [Hyperparameter Tuning Guide](#hyperparameter-tuning)
7. [Common Issues and Solutions](#troubleshooting)

## Introduction to Deep RL and PPO {#introduction}

### What is Deep Reinforcement Learning?
Deep Reinforcement Learning (Deep RL) is a type of AI that learns by **trial and error**:
- The AI (called an "agent") observes the current state
- It chooses an action based on what it thinks is best
- It receives a reward (positive or negative) for that action
- It learns from this experience to make better choices in the future

### What is PPO?
**PPO (Proximal Policy Optimization)** is one of the most successful Deep RL algorithms:
- **Stable**: Doesn't make too big changes at once (that's the "proximal" part)
- **Sample Efficient**: Learns effectively from limited data
- **Versatile**: Works well on many different types of problems
- **Reliable**: Less likely to fail during training than other algorithms

### Why PPO for HVAC Control?
PPO is perfect for building control because:
- **Continuous Actions**: Can set precise temperature setpoints (not just on/off)
- **Multi-Objective**: Can balance energy efficiency and comfort simultaneously
- **Adaptive**: Learns to handle different weather conditions and occupancy patterns
- **Stable Training**: Won't suddenly "forget" what it learned

### The Learning Process
1. **Observation**: AI observes building state (temperature, weather, energy use, etc.)
2. **Action**: AI sets heating and cooling setpoints
3. **Reward**: AI gets positive reward for good energy/comfort balance, negative for bad
4. **Learning**: AI adjusts its decision-making to get better rewards

## Understanding the Code Structure {#code-structure}

### Key Components

#### 1. Environment Wrappers
```python
# These "wrap" the environment to make training work better
env = NormalizeObservation(env)  # Scales observations to similar ranges
env = NormalizeAction(env)       # Scales actions to [-1, 1] range
env = LoggerWrapper(env)         # Records interactions for analysis
env = CSVLogger(env)            # Saves data to CSV files
```

**Why we need wrappers:**
- **NormalizeObservation**: Temperature might be 20-30°C, but energy could be 0-50000W. AI learns better when all numbers are in similar ranges.
- **NormalizeAction**: PPO expects actions in [-1, 1] range, but setpoints are in °C. This converts between them.
- **LoggerWrapper**: Records everything so we can analyze what happened later.

#### 2. PPO Model Configuration
```python
model = PPO(
    policy='MlpPolicy',           # Use neural network for decision making
    env=env,                      # Environment to learn in
    learning_rate=0.0003,         # How fast to learn (smaller = more stable)
    n_steps=2048,                 # Steps to collect before each update
    batch_size=64,                # Size of training batches
    n_epochs=10,                  # How many times to use each batch of data
    gamma=0.99,                   # How much future rewards matter
    verbose=1                     # Print progress
)
```

#### 3. Training Callbacks
```python
eval_callback = EvalCallback(
    eval_env=eval_env,                    # Environment for testing
    best_model_save_path="./models/",     # Where to save best model
    eval_freq=10000,                      # Test every 10,000 steps
    n_eval_episodes=3,                    # Run 3 test episodes each time
    deterministic=True                    # Use best actions (no randomness) for testing
)
```

## Step-by-Step Training Process {#training-process}

### Phase 1: Environment Setup
```python
# 1. Create training environment
train_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="Training")

# 2. Apply wrappers for better training
train_env = NormalizeObservation(train_env)  # Scale observations
train_env = NormalizeAction(train_env)       # Scale actions
train_env = LoggerWrapper(train_env)         # Add logging
train_env = CSVLogger(train_env)            # Save to CSV

# 3. Create evaluation environment (same setup)
eval_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="Evaluation")
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
```

**What happens here:**
- Creates two separate environments: one for training, one for testing
- Applies wrappers to make training more stable and effective
- Sets up logging to track progress

### Phase 2: Model Creation
```python
# Create PPO model with good default settings
model = PPO(
    'MlpPolicy',              # Neural network policy
    train_env,                # Training environment
    learning_rate=0.0003,     # Learning speed
    n_steps=2048,             # Experience buffer size
    batch_size=64,            # Training batch size
    n_epochs=10,              # Training epochs per update
    gamma=0.99,               # Discount factor for future rewards
    gae_lambda=0.95,          # Advantage estimation parameter
    clip_range=0.2,           # PPO clipping parameter
    verbose=1                 # Print training progress
)
```

**Key parameters explained:**
- **learning_rate**: How big steps the AI takes when learning. Too high = unstable, too low = slow learning
- **n_steps**: How much experience to collect before updating the AI brain
- **batch_size**: How much data to process at once during training
- **n_epochs**: How many times to reuse each batch of experience

### Phase 3: Training Execution
```python
# Set up monitoring and model saving
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path="./models/",
    eval_freq=10000,          # Test every 10,000 training steps
    n_eval_episodes=3,        # Run 3 test episodes each time
    deterministic=True,       # Use best actions for testing
    verbose=1
)

# Start training!
model.learn(
    total_timesteps=100000,   # Train for 100,000 steps
    callback=eval_callback,   # Use the monitoring callback
    progress_bar=True         # Show progress bar
)

# Save the final model
model.save("./models/final_model")
```

**What happens during training:**
1. **Experience Collection**: AI interacts with building for `n_steps` (2048) steps
2. **Learning Update**: AI uses this experience to improve its decision-making
3. **Evaluation**: Every 10,000 steps, test the AI on separate episodes
4. **Model Saving**: If evaluation improves, save this version as the "best model"
5. **Repeat**: Continue until `total_timesteps` (100,000) is reached

### Training Output Explanation
During training, you'll see output like this:
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

## Step-by-Step Evaluation Process {#evaluation-process}

### Phase 1: Environment Setup for Evaluation
```python
# Create evaluation environment
eval_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="Final_Evaluation")

# IMPORTANT: Use same wrappers as training, but with fixed parameters
eval_env = NormalizeObservation(
    eval_env, 
    mean=training_mean,           # Use training normalization parameters
    var=training_var,             # Use training normalization parameters
    automatic_update=False        # Don't update these during evaluation
)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
```

**Why fixed normalization parameters?**
- During training, normalization parameters are learned from data
- For fair evaluation, we must use the SAME parameters the model was trained with
- If we use different normalization, the model will perform poorly

### Phase 2: Model Loading
```python
# Load a previously trained model
model = PPO.load("./models/best_model")  # Loads the best model from training

# Or load the final model
model = PPO.load("./models/final_model")  # Loads the final model from training
```

### Phase 3: Evaluation Execution
```python
def evaluate_model(model, eval_env, num_episodes=5):
    """Run evaluation episodes and collect results."""
    
    results = {
        'episode_rewards': [],
        'episode_energies': [],
        'episode_comfort_violations': []
    }
    
    for episode in range(num_episodes):
        print(f"Running evaluation episode {episode + 1}/{num_episodes}")
        
        # Reset environment for new episode
        obs, info = eval_env.reset()
        
        # Initialize tracking variables
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        
        # Run one complete episode
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
        
        print(f"Episode {episode + 1} completed:")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Total energy: {episode_energy:.2f} kWh")
        print(f"  Comfort violations: {episode_comfort_violation:.2f} °C⋅hours")
    
    return results
```

### Phase 4: Results Analysis
```python
# Calculate summary statistics
mean_reward = np.mean(results['episode_rewards'])
std_reward = np.std(results['episode_rewards'])
mean_energy = np.mean(results['episode_energies'])
mean_comfort = np.mean(results['episode_comfort_violations'])

print(f"Evaluation Results:")
print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Mean energy consumption: {mean_energy:.2f} kWh")
print(f"  Mean comfort violations: {mean_comfort:.2f} °C⋅hours")
```

## Code Examples with Line-by-Line Explanations {#code-examples}

### Complete Training Example
```python
import gymnasium as gym
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger

# ============================================================================
# STEP 1: CREATE ENVIRONMENTS
# ============================================================================

# Create training environment
train_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="PPO_Training")
# gym.make() creates the Sinergym environment
# env_name parameter gives it a unique name for file organization

# Apply wrappers to help training
train_env = NormalizeObservation(train_env)
# NormalizeObservation scales all sensor readings to similar ranges
# This helps because AI learns better when inputs are normalized

train_env = NormalizeAction(train_env) 
# NormalizeAction converts actions to [-1, 1] range
# PPO expects actions in this range for optimal performance

train_env = LoggerWrapper(train_env)
# LoggerWrapper records all interactions for later analysis
# Tracks rewards, actions, observations, and environment info

train_env = CSVLogger(train_env)
# CSVLogger saves interaction data to CSV files
# Useful for plotting and analyzing performance later

# Create evaluation environment (same setup)
eval_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="PPO_Evaluation")
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)

# ============================================================================
# STEP 2: CREATE PPO MODEL
# ============================================================================

model = PPO(
    policy='MlpPolicy',           # Use Multi-Layer Perceptron (neural network)
    env=train_env,                # Environment to learn in
    learning_rate=0.0003,         # How fast to learn (3e-4 is good default)
    n_steps=2048,                 # Steps to collect before each update
    batch_size=64,                # Size of training batches
    n_epochs=10,                  # How many times to reuse each batch
    gamma=0.99,                   # Discount factor (how much future matters)
    gae_lambda=0.95,              # Advantage estimation parameter
    clip_range=0.2,               # PPO clipping (prevents too big updates)
    ent_coef=0.01,                # Entropy coefficient (encourages exploration)
    vf_coef=0.5,                  # Value function coefficient
    verbose=1                     # Print training progress
)

# ============================================================================
# STEP 3: SET UP TRAINING MONITORING
# ============================================================================

eval_callback = EvalCallback(
    eval_env=eval_env,                    # Environment for periodic testing
    best_model_save_path="./models/",     # Where to save the best model
    log_path="./logs/",                   # Where to save evaluation logs
    eval_freq=10000,                      # Test every 10,000 training steps
    n_eval_episodes=3,                    # Run 3 test episodes each time
    deterministic=True,                   # Use best actions (no randomness)
    render=False,                         # Don't show visualization
    verbose=1                             # Print evaluation results
)

# ============================================================================
# STEP 4: TRAIN THE MODEL
# ============================================================================

print("Starting PPO training...")
print("This will take several minutes to hours depending on timesteps and computer speed")

model.learn(
    total_timesteps=100000,       # Total number of training steps
    callback=eval_callback,       # Use evaluation callback for monitoring
    progress_bar=True             # Show training progress bar
)

# ============================================================================
# STEP 5: SAVE THE TRAINED MODEL
# ============================================================================

model.save("./models/final_ppo_model")
print("Training completed! Model saved to ./models/final_ppo_model.zip")

# Clean up environments
train_env.close()
eval_env.close()
```

### Complete Evaluation Example
```python
import gymnasium as gym
import numpy as np
import sinergym
from stable_baselines3 import PPO
from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger

# ============================================================================
# STEP 1: LOAD TRAINED MODEL
# ============================================================================

print("Loading trained PPO model...")
model = PPO.load("./models/best_model")  # Load the best model from training
print("Model loaded successfully!")

# ============================================================================
# STEP 2: CREATE EVALUATION ENVIRONMENT
# ============================================================================

# Create evaluation environment
eval_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="Final_Evaluation")

# CRITICAL: Apply same wrappers as training
# If you used different wrappers, the model will perform poorly!
eval_env = NormalizeObservation(eval_env, automatic_update=False)
# automatic_update=False means normalization parameters won't change
# This ensures fair evaluation with same scaling as training

eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)

# ============================================================================
# STEP 3: RUN EVALUATION EPISODES
# ============================================================================

def evaluate_trained_model(model, env, num_episodes=5):
    """
    Evaluate a trained PPO model over multiple episodes.
    
    Args:
        model: Trained PPO model
        env: Evaluation environment
        num_episodes: Number of episodes to run
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"Running {num_episodes} evaluation episodes...")
    
    # Initialize result tracking
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset environment for new episode
        obs, info = env.reset()
        # obs contains current observations (weather, temperature, etc.)
        # info contains additional information from the environment
        
        # Initialize episode tracking
        episode_reward = 0              # Total reward for this episode
        episode_energy = 0              # Total energy consumption
        episode_comfort_violation = 0   # Total comfort violations
        episode_length = 0              # Number of steps in episode
        
        # Run one complete episode
        terminated = truncated = False
        # terminated: episode ended naturally (simulation complete)
        # truncated: episode ended early (time limit, error, etc.)
        
        while not (terminated or truncated):
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            # deterministic=True: use best action (no exploration)
            # action contains [heating_setpoint, cooling_setpoint]
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            # obs: new observation after taking action
            # reward: immediate reward for this action
            # terminated/truncated: episode ending flags
            # info: additional environment information
            
            # Update episode tracking
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            episode_length += 1
            
            # Optional: print progress every 1000 steps
            if episode_length % 1000 == 0:
                print(f"  Step {episode_length}: Reward = {reward:.2f}, "
                      f"Energy = {info.get('total_power_demand', 0):.1f}W")
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violation)
        episode_lengths.append(episode_length)
        
        # Print episode summary
        print(f"Episode {episode + 1} completed:")
        print(f"  Steps: {episode_length}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Total energy: {episode_energy:.2f} kWh")
        print(f"  Comfort violations: {episode_comfort_violation:.2f} °C⋅hours")
    
    # Calculate summary statistics
    results = {
        'episode_rewards': episode_rewards,
        'episode_energies': episode_energies,
        'episode_comfort_violations': episode_comfort_violations,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'std_energy': np.std(episode_energies),
        'mean_comfort_violation': np.mean(episode_comfort_violations),
        'std_comfort_violation': np.std(episode_comfort_violations),
        'mean_episode_length': np.mean(episode_lengths)
    }
    
    return results

# ============================================================================
# STEP 4: RUN EVALUATION AND ANALYZE RESULTS
# ============================================================================

# Run evaluation
results = evaluate_trained_model(model, eval_env, num_episodes=5)

# Print summary results
print(f"\n{'='*60}")
print("EVALUATION SUMMARY")
print(f"{'='*60}")
print(f"Episodes evaluated: 5")
print(f"Mean episode reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"Mean energy consumption: {results['mean_energy']:.2f} ± {results['std_energy']:.2f} kWh")
print(f"Mean comfort violations: {results['mean_comfort_violation']:.2f} ± {results['std_comfort_violation']:.2f} °C⋅hours")
print(f"Mean episode length: {results['mean_episode_length']:.0f} steps")

# ============================================================================
# STEP 5: COMPARE WITH RANDOM POLICY (BASELINE)
# ============================================================================

def compare_with_random_policy(env, num_episodes=3):
    """Compare trained model with random actions (baseline)."""
    print(f"\nRunning {num_episodes} episodes with random actions for comparison...")
    
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        
        terminated = truncated = False
        while not (terminated or truncated):
            # Take random action instead of using trained model
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
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

# Run random policy comparison
random_results = compare_with_random_policy(eval_env, num_episodes=3)

# Compare results
print(f"\n{'='*60}")
print("PERFORMANCE COMPARISON")
print(f"{'='*60}")
print(f"{'Metric':<25} {'Trained PPO':<15} {'Random Policy':<15} {'Improvement'}")
print(f"{'-'*70}")

# Reward comparison (higher is better)
reward_improvement = results['mean_reward'] - random_results['mean_reward']
print(f"{'Mean Reward':<25} {results['mean_reward']:<15.2f} {random_results['mean_reward']:<15.2f} {reward_improvement:+.2f}")

# Energy comparison (lower is better)
energy_improvement = ((random_results['mean_energy'] - results['mean_energy']) / random_results['mean_energy']) * 100
print(f"{'Energy (kWh)':<25} {results['mean_energy']:<15.2f} {random_results['mean_energy']:<15.2f} {energy_improvement:+.1f}%")

# Comfort comparison (lower is better)
comfort_improvement = ((random_results['mean_comfort_violation'] - results['mean_comfort_violation']) / random_results['mean_comfort_violation']) * 100
print(f"{'Comfort Violations':<25} {results['mean_comfort_violation']:<15.2f} {random_results['mean_comfort_violation']:<15.2f} {comfort_improvement:+.1f}%")

# Clean up
eval_env.close()
print(f"\nEvaluation completed!")
```

## Hyperparameter Tuning Guide {#hyperparameter-tuning}

### Key PPO Hyperparameters

#### 1. Learning Rate (`learning_rate`)
**What it does**: Controls how fast the AI learns
```python
# Conservative (stable but slow)
learning_rate=1e-4  # 0.0001

# Standard (good balance)
learning_rate=3e-4  # 0.0003

# Aggressive (fast but potentially unstable)
learning_rate=1e-3  # 0.001
```
**Tuning tips:**
- Start with 3e-4 (default)
- If training is unstable (loss jumps around), reduce to 1e-4
- If training is too slow, increase to 1e-3
- Monitor training curves to see if learning rate is appropriate

#### 2. Number of Steps (`n_steps`)
**What it does**: How much experience to collect before each learning update
```python
# Small (frequent updates, less stable)
n_steps=1024

# Standard (good balance)
n_steps=2048

# Large (less frequent updates, more stable)
n_steps=4096
```
**Tuning tips:**
- Larger values = more stable but slower learning
- Must be compatible with episode length (should be smaller than episode length)
- Good rule: episode_length / 4 ≤ n_steps ≤ episode_length / 2

#### 3. Batch Size (`batch_size`)
**What it does**: Size of data chunks used for training
```python
# Small (more updates, potentially noisier)
batch_size=32

# Standard 
batch_size=64

# Large (fewer updates, more stable)
batch_size=128
```
**Tuning tips:**
- Must be ≤ n_steps
- Larger batch sizes are generally more stable
- Limited by computer memory

#### 4. Number of Epochs (`n_epochs`)
**What it does**: How many times to reuse each batch of experience
```python
# Conservative (less overfitting)
n_epochs=3

# Standard
n_epochs=10

# Aggressive (maximum learning from data)
n_epochs=20
```
**Tuning tips:**
- More epochs = more learning from each batch of data
- Too many epochs can cause overfitting
- Monitor training loss - if it starts increasing, reduce epochs

### Suggested Hyperparameter Combinations

#### For Beginners (Stable Training)
```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=1e-4,     # Conservative learning rate
    n_steps=2048,           # Standard experience buffer
    batch_size=64,          # Standard batch size
    n_epochs=5,             # Conservative epochs
    gamma=0.99,             # Standard discount
    gae_lambda=0.95,        # Standard GAE
    clip_range=0.2,         # Standard clipping
    verbose=1
)
```

#### For Experienced Users (Faster Learning)
```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,     # Standard learning rate
    n_steps=4096,           # Larger experience buffer
    batch_size=128,         # Larger batches
    n_epochs=10,            # Standard epochs
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)
```

#### For Fast Experimentation (Quick Results)
```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=1e-3,     # Fast learning
    n_steps=1024,           # Smaller buffer for quick updates
    batch_size=64,          # Standard batches
    n_epochs=3,             # Few epochs for speed
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)
```

### Training Duration Guidelines

#### Timestep Recommendations
```python
# Quick test (minimal learning)
total_timesteps = 10000      # ~1-2 episodes, just to test setup

# Development (some learning)
total_timesteps = 50000      # ~5-10 episodes, good for debugging

# Standard training (good learning)
total_timesteps = 200000     # ~20-40 episodes, typical for research

# Production training (excellent learning)
total_timesteps = 1000000    # ~100+ episodes, for best performance
```

## Common Issues and Solutions {#troubleshooting}

### Issue 1: Training is Very Slow
**Symptoms**: Training takes forever, very low FPS
**Solutions**:
```python
# Reduce episode length (if possible)
config_params = {'timesteps_per_hour': 4}  # Instead of default 1

# Reduce n_steps
n_steps = 1024  # Instead of 2048

# Use CPU instead of GPU for small models
device = 'cpu'

# Reduce evaluation frequency
eval_freq = 50000  # Instead of 10000
```

### Issue 2: Training is Unstable (Rewards Jump Around)
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

### Issue 3: Model Doesn't Learn (Flat Reward Curves)
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

# Use smaller networks (not shown in basic examples)
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

### Issue 6: Model File Loading Errors
**Symptoms**: Can't load saved models, file not found errors
**Solutions**:
```python
# Check file paths carefully
import os
print(os.path.exists("./models/best_model.zip"))  # Should be True

# Use absolute paths if needed
model_path = os.path.abspath("./models/best_model")

# Ensure .zip extension is handled correctly
model = PPO.load("./models/best_model")  # Don't include .zip
```

### Debugging Checklist
1. **Environment Setup**:
   - [ ] Same wrappers for training and evaluation
   - [ ] Correct normalization parameters
   - [ ] Environment names are unique
   
2. **Model Configuration**:
   - [ ] Reasonable hyperparameters for your setup
   - [ ] Sufficient training timesteps
   - [ ] Appropriate evaluation frequency
   
3. **Resource Management**:
   - [ ] Enough disk space for logs and models
   - [ ] Sufficient RAM for chosen hyperparameters
   - [ ] Environments are closed after use
   
4. **File Management**:
   - [ ] Model save paths exist
   - [ ] Model files are saved correctly
   - [ ] File permissions allow read/write

This guide should give you a comprehensive understanding of PPO training and evaluation for the Sinergym 5Zone environment. Start with the simple examples and gradually experiment with different hyperparameters as you become more comfortable with the process.