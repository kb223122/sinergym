# PPO Training Duration in Sinergym: Timesteps vs Episodes

## Overview

In Sinergym PPO training, there are **two main approaches** to define training duration:

1. **Timesteps-based training** (most common)
2. **Episodes-based training** (converted to timesteps)

Let me explain both approaches in detail.

## 1. Timesteps-Based Training (Primary Method)

### How It Works
Training duration is defined by specifying the total number of **timesteps** (individual environment steps) the agent should train for.

### Key Parameters
```python
TOTAL_TIMESTEPS = 50000    # Total steps to train
EVAL_FREQ = 10000          # Evaluate every 10,000 steps
N_EVAL_EPISODES = 2        # Episodes per evaluation
```

### Implementation Examples

#### Beginner Tutorial (`ppo_beginner_tutorial.py`)
```python
# Training configuration
TOTAL_TIMESTEPS = 50000    # How many steps to train (start small for tutorial)
EVAL_FREQ = 10000          # How often to test the AI during training
N_EVAL_EPISODES = 2        # How many test episodes to run each time

# Training call
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    progress_bar=True
)
```

#### Detailed Guide (`ppo_training_guide_detailed.py`)
```python
class PPOConfig:
    def __init__(self):
        # Training parameters
        self.total_timesteps = 100000      # How many steps to train
        self.eval_freq = 10000             # How often to test during training
        self.n_eval_episodes = 3           # Number of episodes to test each time

# Training call
model.learn(
    total_timesteps=config.total_timesteps,
    callback=callbacks,
    progress_bar=True
)
```

### Advantages of Timesteps-Based Training
- ✅ **Precise control**: Exact number of learning steps
- ✅ **Consistent across environments**: Same timesteps regardless of episode length
- ✅ **Standard in RL**: Most RL algorithms use timesteps
- ✅ **Easy to compare**: Can compare training across different environments

## 2. Episodes-Based Training (Secondary Method)

### How It Works
Training duration is defined by specifying the number of **episodes**, which is then converted to timesteps.

### Key Parameters
```python
EPISODES = 30              # Number of episodes to train
TIMESTEPS_PER_EPISODE = env.get_wrapper_attr('timestep_per_episode')
TOTAL_TIMESTEPS = EPISODES * TIMESTEPS_PER_EPISODE
```

### Implementation Examples

#### Simple Training (`train_agent_simple.py`)
```python
config = {
    'environment': 'Eplus-5zone-hot-continuous-v1',
    'episodes': 30,  # Define episodes first
    # ... other config
}

# Calculate timesteps from episodes
timesteps = config['episodes'] * env.get_wrapper_attr('timestep_per_episode')
print(f"⏱️  Training for {timesteps:,} timesteps ({config['episodes']} episodes)")

# Training call
model.learn(
    total_timesteps=timesteps,
    callback=callback,
    log_interval=config['algorithm']['log_interval']
)
```

#### Official Match (`train_agent_official_match.py`)
```python
config = {
    'environment': 'Eplus-5zone-hot-continuous-v1',
    'episodes': 5,  # OFFICIAL: 5 episodes
    'evaluation': {
        'eval_length': 1,    # OFFICIAL: 1 episode
        'eval_freq': 2       # OFFICIAL: every 2 episodes
    }
}

# Calculate timesteps
timesteps = config['episodes'] * env.get_wrapper_attr('timestep_per_episode')
print(f"⏱️  Training for {timesteps:,} timesteps ({config['episodes']} episodes)")
```

### Advantages of Episodes-Based Training
- ✅ **Intuitive**: Easy to understand (e.g., "train for 30 episodes")
- ✅ **Environment-aware**: Automatically adapts to different episode lengths
- ✅ **Natural units**: Episodes are natural units for building simulation

## 3. Episode Length in Sinergym

### How Episode Length is Determined
```python
# Get episode length from environment
episode_length = env.get_wrapper_attr('timestep_per_episode')
print(f"Episode length: {episode_length} timesteps")
```

### Typical Episode Lengths
- **5Zone environments**: ~8760 timesteps (1 year simulation)
- **Demo environments**: Variable (shorter for testing)
- **Custom environments**: Configurable

### Example Calculations
```python
# For 5Zone environment with 8760 timesteps per episode:
episodes = 30
timesteps_per_episode = 8760
total_timesteps = 30 * 8760 = 262,800 timesteps

# For shorter training:
episodes = 5
total_timesteps = 5 * 8760 = 43,800 timesteps
```

## 4. Training Duration Recommendations

### For Beginners
```python
# Start small for testing
TOTAL_TIMESTEPS = 5000     # Quick test
TOTAL_TIMESTEPS = 50000    # Beginner training
```

### For Serious Training
```python
# Longer training for better performance
TOTAL_TIMESTEPS = 100000   # Standard training
TOTAL_TIMESTEPS = 500000   # Extended training
TOTAL_TIMESTEPS = 1000000  # Comprehensive training
```

### For Episodes-Based
```python
# Short training
EPISODES = 5               # Quick test (5 episodes)

# Standard training
EPISODES = 30              # Moderate training (30 episodes)

# Extended training
EPISODES = 100             # Comprehensive training (100 episodes)
```

## 5. Evaluation Frequency

### Timesteps-Based Evaluation
```python
EVAL_FREQ = 10000          # Evaluate every 10,000 steps
EVAL_FREQ = 50000          # Evaluate every 50,000 steps
```

### Episodes-Based Evaluation
```python
eval_freq_episodes = 2     # Evaluate every 2 episodes
eval_freq_episodes = 5     # Evaluate every 5 episodes
```

## 6. Practical Examples

### Quick Test (Timesteps)
```python
TOTAL_TIMESTEPS = 5000
EVAL_FREQ = 1000
N_EVAL_EPISODES = 1
```

### Standard Training (Timesteps)
```python
TOTAL_TIMESTEPS = 100000
EVAL_FREQ = 10000
N_EVAL_EPISODES = 3
```

### Quick Test (Episodes)
```python
EPISODES = 5
eval_freq_episodes = 1
eval_length = 1
```

### Standard Training (Episodes)
```python
EPISODES = 30
eval_freq_episodes = 2
eval_length = 3
```

## 7. Conversion Between Methods

### Episodes to Timesteps
```python
def episodes_to_timesteps(episodes, env):
    timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
    return episodes * timesteps_per_episode

# Example
episodes = 30
timesteps = episodes_to_timesteps(episodes, env)
print(f"{episodes} episodes = {timesteps:,} timesteps")
```

### Timesteps to Episodes
```python
def timesteps_to_episodes(timesteps, env):
    timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
    return timesteps / timesteps_per_episode

# Example
timesteps = 100000
episodes = timesteps_to_episodes(timesteps, env)
print(f"{timesteps:,} timesteps ≈ {episodes:.1f} episodes")
```

## 8. Best Practices

### Choose Based on Your Needs
- **Use timesteps** if you want precise control and consistency
- **Use episodes** if you want intuitive, environment-aware training

### Start Small
- Begin with small training durations to test your setup
- Gradually increase as you understand the environment

### Monitor Progress
- Use evaluation callbacks to track performance
- Adjust training duration based on learning curves

### Consider Environment
- Longer episodes (like 1-year simulations) may need more training
- Shorter episodes can train faster but may not capture seasonal patterns

## Summary

**Primary Method**: Use `total_timesteps` for precise control
**Secondary Method**: Use episodes and convert to timesteps for intuitive training

Both methods ultimately use timesteps internally, but episodes provide a more intuitive way to think about training duration in building simulation contexts.