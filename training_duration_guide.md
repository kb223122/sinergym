# PPO Training Duration and Workflow Best Practices

## Training Duration Recommendations

### Quick Reference Table

| Purpose | Timesteps | Episodes | Training Time | Use Case |
|---------|-----------|----------|---------------|----------|
| **Quick Test** | 50,000 | ~1-2 | 30 min | Setup testing |
| **Development** | 200,000 | ~6 | 2-3 hours | Debugging/prototyping |
| **Standard Training** | 500,000 | ~14 | 6-8 hours | Research/experiments |
| **Production Quality** | 1,000,000 | ~28 | 12-15 hours | Best performance |
| **High Performance** | 2,000,000+ | ~57+ | 24+ hours | Competition/deployment |

### Detailed Recommendations

#### 1. **Quick Testing (50,000 timesteps)**
```python
total_timesteps = 50000  # ~1.5 episodes
```
- **Purpose**: Verify setup works, debug code
- **Expectation**: Minimal learning, just ensures no crashes
- **Time**: 30-60 minutes

#### 2. **Development Training (200,000 timesteps)**
```python
total_timesteps = 200000  # ~6 episodes
```
- **Purpose**: Initial development, hyperparameter testing
- **Expectation**: Some learning visible, basic patterns
- **Time**: 2-3 hours

#### 3. **Standard Training (500,000 timesteps)**
```python
total_timesteps = 500000  # ~14 episodes
```
- **Purpose**: Research experiments, good baseline
- **Expectation**: Clear learning, decent performance
- **Time**: 6-8 hours

#### 4. **Production Training (1,000,000+ timesteps)**
```python
total_timesteps = 1000000  # ~28 episodes
```
- **Purpose**: Best performance, deployment-ready
- **Expectation**: Excellent performance, stable policy
- **Time**: 12-15 hours

## Best Practice: Separate Training and Evaluation

**YES - Always run training and evaluation separately!**

### Recommended Workflow

#### Phase 1: Training (Run Once)
```python
# File: train_ppo_agent.py
import gymnasium as gym
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction, LoggerWrapper

def train_ppo_model():
    # 1. Create training environment
    train_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="PPO_Training")
    train_env = NormalizeObservation(train_env)
    train_env = NormalizeAction(train_env)
    train_env = LoggerWrapper(train_env)
    
    # 2. Create evaluation environment for monitoring during training
    eval_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="PPO_Training_Monitor")
    eval_env = NormalizeObservation(eval_env)
    eval_env = NormalizeAction(eval_env)
    eval_env = LoggerWrapper(eval_env)
    
    # 3. Create PPO model
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1
    )
    
    # 4. Set up monitoring (NOT final evaluation)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=50000,          # Check every 50k timesteps
        n_eval_episodes=1,        # Just 1 episode for monitoring
        deterministic=True,
        verbose=1
    )
    
    # 5. TRAIN THE MODEL
    print("Starting PPO training...")
    print(f"Training for 1,000,000 timesteps (~28 episodes)")
    
    model.learn(
        total_timesteps=1000000,  # Substantial training
        callback=eval_callback,
        progress_bar=True
    )
    
    # 6. Save final model
    model.save("./models/final_ppo_model")
    print("Training completed!")
    
    # 7. Clean up
    train_env.close()
    eval_env.close()
    
    return model

if __name__ == "__main__":
    train_ppo_model()
```

#### Phase 2: Comprehensive Evaluation (Run After Training)
```python
# File: evaluate_ppo_agent.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction, LoggerWrapper

def evaluate_trained_model(model_path, num_eval_episodes=5):
    """
    Comprehensive evaluation of trained PPO model.
    
    Args:
        model_path: Path to saved model
        num_eval_episodes: Number of episodes to evaluate (recommend 3-5)
    """
    
    # 1. Load trained model
    print(f"Loading trained model from: {model_path}")
    model = PPO.load(model_path)
    
    # 2. Create evaluation environment
    # IMPORTANT: Use same wrappers as training!
    eval_env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="Final_Evaluation")
    eval_env = NormalizeObservation(eval_env, automatic_update=False)  # Fixed normalization
    eval_env = NormalizeAction(eval_env)
    eval_env = LoggerWrapper(eval_env)
    
    # 3. Run evaluation episodes
    print(f"Running {num_eval_episodes} evaluation episodes...")
    print("Each episode = 1 full year simulation")
    
    results = []
    
    for episode in range(num_eval_episodes):
        print(f"\n--- Evaluation Episode {episode + 1}/{num_eval_episodes} ---")
        
        # Reset for new episode
        obs, info = eval_env.reset()
        
        # Track metrics
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        step_count = 0
        
        # Run complete episode (1 year)
        terminated = truncated = False
        while not (terminated or truncated):
            # Get action from trained model (deterministic = best action)
            action, _ = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            step_count += 1
            
            # Progress update every month (~2920 steps)
            if step_count % 2920 == 0:
                month = step_count // 2920
                print(f"  Month {month}: Reward = {reward:.2f}")
        
        # Store episode results
        result = {
            'episode': episode + 1,
            'total_reward': episode_reward,
            'total_energy_kwh': episode_energy,
            'total_comfort_violations': episode_comfort_violation,
            'total_steps': step_count
        }
        results.append(result)
        
        # Print episode summary
        print(f"Episode {episode + 1} completed:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Total energy: {episode_energy:.2f} kWh")
        print(f"  Comfort violations: {episode_comfort_violation:.2f} °C⋅hours")
    
    # 4. Calculate summary statistics
    rewards = [r['total_reward'] for r in results]
    energies = [r['total_energy_kwh'] for r in results]
    comforts = [r['total_comfort_violations'] for r in results]
    
    summary = {
        'num_episodes': num_eval_episodes,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_energy': np.mean(energies),
        'std_energy': np.std(energies),
        'mean_comfort': np.mean(comforts),
        'std_comfort': np.std(comforts),
        'detailed_results': results
    }
    
    # 5. Print final summary
    print(f"\n{'='*60}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes evaluated: {num_eval_episodes}")
    print(f"Mean reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Mean energy: {summary['mean_energy']:.2f} ± {summary['std_energy']:.2f} kWh")
    print(f"Mean comfort violations: {summary['mean_comfort']:.2f} ± {summary['std_comfort']:.2f} °C⋅hours")
    
    eval_env.close()
    return summary

def compare_with_baseline(model_path, num_episodes=3):
    """Compare trained model with random policy baseline."""
    
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON")
    print(f"{'='*60}")
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make('Eplus-5zone-hot-continuous-v1', env_name="Baseline_Comparison")
    env = NormalizeObservation(env, automatic_update=False)
    env = NormalizeAction(env)
    env = LoggerWrapper(env)
    
    # Test trained model
    print(f"Testing trained model ({num_episodes} episodes)...")
    trained_rewards = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
        trained_rewards.append(episode_reward)
        print(f"  Trained Episode {episode + 1}: {episode_reward:.2f}")
    
    # Test random policy
    print(f"\nTesting random policy ({num_episodes} episodes)...")
    random_rewards = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
        random_rewards.append(episode_reward)
        print(f"  Random Episode {episode + 1}: {episode_reward:.2f}")
    
    # Compare results
    trained_mean = np.mean(trained_rewards)
    random_mean = np.mean(random_rewards)
    improvement = trained_mean - random_mean
    
    print(f"\nComparison Results:")
    print(f"Trained Policy: {trained_mean:.2f}")
    print(f"Random Policy:  {random_mean:.2f}")
    print(f"Improvement:    {improvement:.2f} ({improvement/abs(random_mean)*100:.1f}%)")
    
    env.close()
    return trained_mean, random_mean

if __name__ == "__main__":
    # Evaluate the trained model
    results = evaluate_trained_model("./models/best_model", num_eval_episodes=5)
    
    # Compare with baseline
    compare_with_baseline("./models/best_model", num_episodes=3)
```

## Why This Workflow is Best Practice

### ✅ **Advantages of Separate Training/Evaluation:**

1. **No Data Leakage**: Evaluation uses completely fresh episodes
2. **Statistical Validity**: Multiple episodes give reliable performance estimates
3. **Resource Efficiency**: Training can run uninterrupted
4. **Clean Results**: Clear separation between learning and testing
5. **Reproducibility**: Easy to reproduce evaluation results

### ❌ **Problems with Combined Training/Evaluation:**

1. **Overfitting**: Model might memorize specific episodes
2. **Biased Results**: Evaluation on episodes the model trained on
3. **Inefficiency**: Constant interruption of training
4. **Unreliable Metrics**: Performance estimates are inflated

## Specific Answers to Your Questions

### Q1: How many episodes/timesteps for training?

**Recommended for 5Zone environment:**
- **Minimum viable**: 500,000 timesteps (~14 episodes)
- **Good performance**: 1,000,000 timesteps (~28 episodes)  
- **Excellent performance**: 2,000,000 timesteps (~57 episodes)

### Q2: Should I run training and evaluation separately?

**YES! Always separate training and evaluation:**

1. **Train once**: 1,000,000 timesteps, save best model
2. **Evaluate separately**: Load saved model, test on 3-5 fresh episodes

### Q3: Is 1 episode enough for evaluation?

**NO - Use 3-5 episodes for evaluation:**
- 1 episode = 1 data point (not statistically reliable)
- 3-5 episodes = enough for meaningful statistics
- More episodes = better statistics but diminishing returns

## Complete Usage Example

```bash
# Step 1: Train the model (run once)
python train_ppo_agent.py
# Output: ./models/best_model.zip, ./models/final_ppo_model.zip

# Step 2: Evaluate the trained model (run after training)
python evaluate_ppo_agent.py
# Output: Comprehensive performance statistics

# Step 3: Use for deployment or further analysis
```

## Training Progress Monitoring

During training, watch these metrics:
```
| rollout/ep_rew_mean    | -45.6  |  # Should improve over time
| eval/mean_reward       | -42.1  |  # Should match or exceed rollout
| train/learning_rate    | 0.0003 |  # Should match your setting
| time/fps              | 23.4   |  # Training speed
| time/total_timesteps  | 450000 |  # Progress toward goal
```

**Good signs:**
- Rewards gradually improve (become less negative)
- Evaluation performance tracks training performance
- Stable training (no sudden crashes or huge jumps)

This workflow ensures you get reliable, unbiased performance estimates while maximizing training efficiency!