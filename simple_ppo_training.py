#!/usr/bin/env python3
"""
Simple PPO Training for 5Zone Environment
=========================================

Easy-to-use PPO training script with customizable reward weights and run periods.
Uses the environment_config utility for easy configuration.

Usage:
    python simple_ppo_training.py
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Import configuration utility
from environment_config import get_config, print_config

print("🚀 Simple PPO Training for 5Zone Environment")
print("=" * 60)

try:
    import gymnasium as gym
    import sinergym
    from sinergym.utils.rewards import LinearReward
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install sinergym[drl] stable-baselines3 gymnasium")
    sys.exit(1)

def create_custom_reward_function(energy_weight, lambda_energy, lambda_temperature):
    """Create custom reward function."""
    return LinearReward(
        temperature_variables=['air_temperature'],
        energy_variables=['HVAC_electricity_demand_rate'],
        range_comfort_winter=(20.0, 23.5),
        range_comfort_summer=(23.0, 26.0),
        energy_weight=energy_weight,
        lambda_energy=lambda_energy,
        lambda_temperature=lambda_temperature
    )

def create_run_period_config(timestep_per_hour, start_month, start_day, end_month, end_day):
    """Create run period configuration."""
    runperiod = (start_month, start_day, end_month, end_day)
    
    # Calculate timesteps per episode
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if start_month == end_month:
        total_days = end_day - start_day + 1
    else:
        total_days = (days_per_month[start_month-1] - start_day + 1)
        for month in range(start_month + 1, end_month):
            total_days += days_per_month[month-1]
        total_days += end_day
    
    timesteps_per_episode = total_days * 24 // timestep_per_hour
    
    return {
        'timestep_per_hour': timestep_per_hour,
        'runperiod': runperiod,
        'timesteps_per_episode': timesteps_per_episode
    }

def create_environment(reward_config, run_period_config):
    """Create and configure environment."""
    
    print(f"\n🔧 Creating Environment")
    print("-" * 40)
    
    try:
        # Create base environment
        env = gym.make('Eplus-5zone-hot-continuous-v1')
        
        # Apply custom reward function
        custom_reward = create_custom_reward_function(
            reward_config['energy_weight'],
            reward_config['lambda_energy'],
            reward_config['lambda_temperature']
        )
        env.reward_fn = custom_reward
        
        # Apply custom run period
        env.timestep_per_hour = run_period_config['timestep_per_hour']
        env.runperiod = run_period_config['runperiod']
        
        print("✅ Environment created successfully")
        return env
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return None

def train_ppo(env, eval_env, config):
    """Train PPO agent."""
    
    print(f"\n🎯 Starting PPO Training")
    print("-" * 40)
    print(f"Total Timesteps: {config['training']['total_timesteps']:,}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Timesteps per Episode: {env.timestep_per_episode:,}")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['training']['learning_rate'],
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        verbose=1
    )
    
    # Setup callbacks
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)
    
    callbacks = [
        CheckpointCallback(
            save_freq=10000,
            save_path="./models/",
            name_prefix="ppo_5zone"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path="./models/",
            log_path="./logs/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
    ]
    
    # Train
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        
        return model
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

def evaluate_model(model, env, num_episodes=3):
    """Evaluate trained model."""
    
    print(f"\n📊 Evaluating Model ({num_episodes} episodes)")
    print("-" * 40)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.4f}, Length = {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    
    return mean_reward

def main():
    """Main training function."""
    
    # =============================================================================
    # CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
    # =============================================================================
    
    # Choose your configurations here:
    reward_config_name = 'energy_focused'      # Options: default, energy_focused, comfort_focused, extreme_energy, extreme_comfort, custom
    run_period_config_name = 'summer_only'     # Options: default, summer_only, winter_only, spring_only, two_hour_timesteps, four_hour_timesteps, summer_two_hour
    training_config_name = 'quick_test'        # Options: default, quick_test, long_training, high_lr, low_lr
    
    # Get configuration
    config = get_config(reward_config_name, run_period_config_name, training_config_name)
    
    # Print configuration
    print_config(config)
    
    # =============================================================================
    # CREATE ENVIRONMENTS
    # =============================================================================
    
    # Create training environment
    train_env = create_environment(config['reward'], config['run_period'])
    if train_env is None:
        print("❌ Failed to create training environment")
        return
    
    # Create evaluation environment
    eval_env = create_environment(config['reward'], config['run_period'])
    if eval_env is None:
        print("❌ Failed to create evaluation environment")
        train_env.close()
        return
    
    # =============================================================================
    # TRAIN AND EVALUATE
    # =============================================================================
    
    try:
        # Train PPO agent
        model = train_ppo(train_env, eval_env, config)
        
        if model is None:
            print("❌ Training failed")
            train_env.close()
            eval_env.close()
            return
        
        # Evaluate model
        mean_reward = evaluate_model(model, eval_env, num_episodes=config['training']['eval_episodes'])
        
        # Save model
        model_path = f"./models/ppo_5zone_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Final summary
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Configuration: {reward_config_name} + {run_period_config_name} + {training_config_name}")
        print(f"Final Mean Reward: {mean_reward:.4f}")
        print(f"Model saved to: {model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    main()