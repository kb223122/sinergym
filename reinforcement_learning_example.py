#!/usr/bin/env python3
"""
Reinforcement Learning Example with Sinergym
This script demonstrates how to train a reinforcement learning agent using Stable Baselines 3.
"""

import gymnasium as gym
import numpy as np
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sinergym.utils.wrappers import (
    NormalizeAction,
    NormalizeObservation,
    LoggerWrapper,
    CSVLogger
)

def create_env(env_name, use_wrappers=True):
    """
    Create and configure environment with optional wrappers.
    
    Args:
        env_name (str): Name of the environment to create
        use_wrappers (bool): Whether to apply wrappers for RL training
        
    Returns:
        gym.Env: Configured environment
    """
    env = gym.make(env_name)
    
    if use_wrappers:
        # Apply wrappers for better RL training
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        env = LoggerWrapper(env)
        env = CSVLogger(env)
    
    return env

def train_agent(env, total_timesteps=50000, eval_freq=1000):
    """
    Train a PPO agent on the given environment.
    
    Args:
        env: Training environment
        total_timesteps (int): Total timesteps for training
        eval_freq (int): Frequency of evaluation
        
    Returns:
        PPO: Trained model
    """
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    
    # Create evaluation environment
    eval_env = create_env(env.unwrapped.spec.id, use_wrappers=True)
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    eval_env.close()
    return model

def evaluate_model(model, env, num_episodes=3):
    """
    Evaluate a trained model on the given environment.
    
    Args:
        model: Trained model
        env: Evaluation environment
        num_episodes (int): Number of episodes to evaluate
        
    Returns:
        dict: Evaluation results
    """
    print(f"Evaluating model over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        step_count = 0
        
        terminated = truncated = False
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violation)
        
        print(f"    Steps: {step_count}, Reward: {episode_reward:.2f}, "
              f"Energy: {episode_energy:.2f} kWh, "
              f"Comfort violations: {episode_comfort_violation:.2f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'std_energy': np.std(episode_energies),
        'mean_comfort_violation': np.mean(episode_comfort_violations),
        'std_comfort_violation': np.std(episode_comfort_violations),
        'episode_rewards': episode_rewards,
        'episode_energies': episode_energies,
        'episode_comfort_violations': episode_comfort_violations
    }

def compare_with_random(env, num_episodes=3):
    """
    Compare trained model performance with random actions.
    
    Args:
        env: Environment to test on
        num_episodes (int): Number of episodes to evaluate
        
    Returns:
        dict: Random policy results
    """
    print(f"Evaluating random policy over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        step_count = 0
        
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violation)
        
        print(f"    Steps: {step_count}, Reward: {episode_reward:.2f}, "
              f"Energy: {episode_energy:.2f} kWh, "
              f"Comfort violations: {episode_comfort_violation:.2f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'std_energy': np.std(episode_energies),
        'mean_comfort_violation': np.mean(episode_comfort_violations),
        'std_comfort_violation': np.std(episode_comfort_violations),
        'episode_rewards': episode_rewards,
        'episode_energies': episode_energies,
        'episode_comfort_violations': episode_comfort_violations
    }

def main():
    print("=== Reinforcement Learning Example with Sinergym ===\n")
    
    # Configuration
    env_name = 'Eplus-5zone-hot-continuous-stochastic-v1'
    total_timesteps = 50000  # Adjust based on your needs
    eval_freq = 5000
    
    print(f"Environment: {env_name}")
    print(f"Training timesteps: {total_timesteps}")
    print(f"Evaluation frequency: {eval_freq}\n")
    
    # Create training environment
    print("1. Creating training environment...")
    train_env = create_env(env_name, use_wrappers=True)
    print(f"   Environment created: {train_env.name}")
    print(f"   Observation space: {train_env.observation_space}")
    print(f"   Action space: {train_env.action_space}")
    
    # Train the agent
    print("\n2. Training PPO agent...")
    model = train_agent(train_env, total_timesteps, eval_freq)
    
    # Save the model
    model_path = "ppo_sinergym_model"
    model.save(model_path)
    print(f"   Model saved to: {model_path}")
    
    # Create evaluation environment
    print("\n3. Creating evaluation environment...")
    eval_env = create_env(env_name, use_wrappers=True)
    
    # Evaluate trained model
    print("\n4. Evaluating trained model...")
    trained_results = evaluate_model(model, eval_env, num_episodes=3)
    
    # Compare with random policy
    print("\n5. Comparing with random policy...")
    random_results = compare_with_random(eval_env, num_episodes=3)
    
    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<20} {'Trained Model':<15} {'Random Policy':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Reward comparison
    reward_improvement = ((random_results['mean_reward'] - trained_results['mean_reward']) / 
                         abs(random_results['mean_reward'])) * 100
    print(f"{'Mean Reward':<20} {trained_results['mean_reward']:<15.2f} "
          f"{random_results['mean_reward']:<15.2f} {reward_improvement:<15.1f}%")
    
    # Energy comparison
    energy_improvement = ((random_results['mean_energy'] - trained_results['mean_energy']) / 
                         random_results['mean_energy']) * 100
    print(f"{'Mean Energy (kWh)':<20} {trained_results['mean_energy']:<15.2f} "
          f"{random_results['mean_energy']:<15.2f} {energy_improvement:<15.1f}%")
    
    # Comfort comparison
    comfort_improvement = ((random_results['mean_comfort_violation'] - trained_results['mean_comfort_violation']) / 
                          random_results['mean_comfort_violation']) * 100
    print(f"{'Mean Comfort Viol.':<20} {trained_results['mean_comfort_violation']:<15.2f} "
          f"{random_results['mean_comfort_violation']:<15.2f} {comfort_improvement:<15.1f}%")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    print(f"\nTraining completed! Check the following directories for outputs:")
    print(f"  - Best model: ./best_model/")
    print(f"  - Training logs: ./logs/")
    print(f"  - Episode data: {eval_env.episode_path}")

if __name__ == "__main__":
    main()