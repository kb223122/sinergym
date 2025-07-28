#!/usr/bin/env python3
"""
Direct PPO Agent Training for 5Zone Environment
===============================================

This script provides a complete PPO (Proximal Policy Optimization) training implementation
for the 5Zone building control environment. Everything is self-contained with no external
config files required.

The 5Zone environment simulates a 5-zone commercial building with HVAC control.
The PPO agent learns to control heating and cooling setpoints to balance energy
efficiency with occupant comfort.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import logging

# Import Sinergym components
from sinergym.envs import EplusEnv
from sinergym.utils.rewards import LinearReward
from sinergym.utils.constants import *

# Import Stable-Baselines3 for PPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    StopTrainingOnRewardThreshold,
    BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOTrainingConfig:
    """Configuration class for PPO training parameters."""
    
    def __init__(self):
        # Environment settings
        self.building_file = "5ZoneAutoDXVAV.epJSON"
        self.weather_file = "USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw"  # Hot weather
        self.experiment_name = f"ppo_5zone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Training parameters
        self.total_timesteps = 100000  # Total training steps
        self.eval_freq = 5000  # Evaluate every N steps
        self.save_freq = 10000  # Save model every N steps
        self.n_eval_episodes = 5  # Number of episodes for evaluation
        
        # PPO hyperparameters
        self.learning_rate = 3e-4
        self.n_steps = 2048  # Steps per update
        self.batch_size = 64
        self.n_epochs = 10  # Number of epochs per update
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE lambda parameter
        self.clip_range = 0.2  # PPO clip range
        self.clip_range_vf = None  # Value function clip range
        self.ent_coef = 0.01  # Entropy coefficient
        self.vf_coef = 0.5  # Value function coefficient
        self.max_grad_norm = 0.5  # Maximum gradient norm
        self.target_kl = None  # Target KL divergence
        
        # Network architecture
        self.policy_kwargs = {
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
            "activation_fn": "tanh"
        }
        
        # Environment parameters
        self.config_params = {
            "runperiod": (1, 1, 1991, 1, 31, 1991),  # January 1991
            "timesteps_per_hour": 1
        }

def create_5zone_environment(
    weather_file: str = "USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw",
    config_params: Optional[Dict] = None,
    is_eval: bool = False
) -> EplusEnv:
    """
    Create a 5Zone environment with the specified configuration.
    
    Args:
        weather_file: Weather file to use
        config_params: Environment configuration parameters
        is_eval: Whether this is for evaluation (affects logging)
    
    Returns:
        Configured 5Zone environment
    """
    
    # Default configuration parameters
    if config_params is None:
        config_params = {
            "runperiod": (1, 1, 1991, 1, 31, 1991),
            "timesteps_per_hour": 1
        }
    
    # Define the environment parameters based on 5Zone configuration
    env_kwargs = {
        # Building and weather
        "building_file": "5ZoneAutoDXVAV.epJSON",
        "weather_files": weather_file,
        
        # Time variables
        "time_variables": [
            "month",
            "day_of_month", 
            "hour"
        ],
        
        # Observation variables (output variables)
        "variables": {
            "outdoor_temperature": (
                "Site Outdoor Air DryBulb Temperature",
                "Environment"
            ),
            "outdoor_humidity": (
                "Site Outdoor Air Relative Humidity", 
                "Environment"
            ),
            "wind_speed": (
                "Site Wind Speed",
                "Environment"
            ),
            "wind_direction": (
                "Site Wind Direction",
                "Environment"
            ),
            "diffuse_solar_radiation": (
                "Site Diffuse Solar Radiation Rate per Area",
                "Environment"
            ),
            "direct_solar_radiation": (
                "Site Direct Solar Radiation Rate per Area",
                "Environment"
            ),
            "htg_setpoint": (
                "Zone Thermostat Heating Setpoint Temperature",
                "SPACE5-1"
            ),
            "clg_setpoint": (
                "Zone Thermostat Cooling Setpoint Temperature",
                "SPACE5-1"
            ),
            "air_temperature": (
                "Zone Air Temperature",
                "SPACE5-1"
            ),
            "air_humidity": (
                "Zone Air Relative Humidity",
                "SPACE5-1"
            ),
            "people_occupant": (
                "Zone People Occupant Count",
                "SPACE5-1"
            ),
            "co2_emission": (
                "Environmental Impact Total CO2 Emissions Carbon Equivalent Mass",
                "site"
            ),
            "HVAC_electricity_demand_rate": (
                "Facility Total HVAC Electricity Demand Rate",
                "Whole Building"
            )
        },
        
        # Meters
        "meters": {
            "Electricity:HVAC": "total_electricity_HVAC"
        },
        
        # Action variables (actuators)
        "actuators": {
            "Heating_Setpoint_RL": (
                "Schedule:Compact",
                "Schedule Value",
                "HTG-SETP-SCH"
            ),
            "Cooling_Setpoint_RL": (
                "Schedule:Compact", 
                "Schedule Value",
                "CLG-SETP-SCH"
            )
        },
        
        # Action space (heating and cooling setpoints)
        "action_space": gym.spaces.Box(
            low=np.array([12.0, 23.25], dtype=np.float32),
            high=np.array([23.25, 30.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        ),
        
        # Reward function
        "reward": LinearReward,
        "reward_kwargs": {
            "temperature_variables": ["air_temperature"],
            "energy_variables": ["HVAC_electricity_demand_rate"],
            "range_comfort_winter": (20.0, 23.5),
            "range_comfort_summer": (23.0, 26.0),
            "summer_start": (6, 1),
            "summer_final": (9, 30),
            "energy_weight": 0.5,
            "lambda_energy": 1.0e-4,
            "lambda_temperature": 1.0
        },
        
        # Environment configuration
        "config_params": config_params,
        
        # Environment name
        "env_name": "5zone-training" if not is_eval else "5zone-eval"
    }
    
    # Create the environment
    env = EplusEnv(**env_kwargs)
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    print(f"✅ 5Zone Environment created successfully")
    print(f"   - Building: {env_kwargs['building_file']}")
    print(f"   - Weather: {weather_file}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Action range: Heating [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}°C]")
    print(f"   - Action range: Cooling [{env.action_space.low[1]:.1f}, {env.action_space.high[1]:.1f}°C]")
    
    return env

def create_ppo_model(
    env: EplusEnv,
    config: PPOTrainingConfig
) -> PPO:
    """
    Create a PPO model with the specified configuration.
    
    Args:
        env: Training environment
        config: Training configuration
    
    Returns:
        Configured PPO model
    """
    
    print(f"🤖 Creating PPO model with configuration:")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - N steps: {config.n_steps}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - N epochs: {config.n_epochs}")
    print(f"   - Gamma: {config.gamma}")
    print(f"   - Clip range: {config.clip_range}")
    print(f"   - Entropy coefficient: {config.ent_coef}")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        policy_kwargs=config.policy_kwargs,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{config.experiment_name}/"
    )
    
    return model

def setup_training_callbacks(
    config: PPOTrainingConfig,
    eval_env: EplusEnv
) -> list:
    """
    Setup training callbacks for monitoring and evaluation.
    
    Args:
        config: Training configuration
        eval_env: Evaluation environment
    
    Returns:
        List of callbacks
    """
    
    callbacks = []
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{config.experiment_name}/",
        log_path=f"./logs/{config.experiment_name}/",
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=f"./checkpoints/{config.experiment_name}/",
        name_prefix="ppo_model"
    )
    callbacks.append(checkpoint_callback)
    
    # Create stop training callback when reward threshold is reached
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=1000,  # High reward threshold
        verbose=1
    )
    callbacks.append(stop_callback)
    
    return callbacks

def train_ppo_agent(
    config: PPOTrainingConfig,
    train_env: EplusEnv,
    eval_env: EplusEnv
) -> PPO:
    """
    Train a PPO agent on the 5Zone environment.
    
    Args:
        config: Training configuration
        train_env: Training environment
        eval_env: Evaluation environment
    
    Returns:
        Trained PPO model
    """
    
    print(f"🚀 Starting PPO training for {config.total_timesteps} timesteps")
    print(f"   - Experiment name: {config.experiment_name}")
    print(f"   - Evaluation frequency: {config.eval_freq} steps")
    print(f"   - Save frequency: {config.save_freq} steps")
    
    # Create PPO model
    model = create_ppo_model(train_env, config)
    
    # Setup callbacks
    callbacks = setup_training_callbacks(config, eval_env)
    
    # Create directories
    os.makedirs(f"./models/{config.experiment_name}/", exist_ok=True)
    os.makedirs(f"./checkpoints/{config.experiment_name}/", exist_ok=True)
    os.makedirs(f"./logs/{config.experiment_name}/", exist_ok=True)
    os.makedirs(f"./tensorboard_logs/{config.experiment_name}/", exist_ok=True)
    
    # Train the model
    print("🎯 Training started...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"./models/{config.experiment_name}/final_model"
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")
    
    return model

def evaluate_trained_model(
    model: PPO,
    eval_env: EplusEnv,
    n_episodes: int = 10
) -> Dict[str, float]:
    """
    Evaluate a trained PPO model.
    
    Args:
        model: Trained PPO model
        eval_env: Evaluation environment
        n_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print(f"📊 Evaluating model over {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    episode_energy_consumption = []
    episode_comfort_violations = []
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        episode_energy = 0
        episode_comfort_violations_count = 0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Extract energy consumption and comfort violations from info
            if 'HVAC_electricity_demand_rate' in info:
                episode_energy += info['HVAC_electricity_demand_rate']
            
            if 'air_temperature' in info:
                temp = info['air_temperature']
                # Check comfort violations (outside 20-26°C range)
                if temp < 20.0 or temp > 26.0:
                    episode_comfort_violations_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_energy_consumption.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violations_count)
        
        print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Length={episode_length}, Energy={episode_energy:.2f}, "
              f"Comfort violations={episode_comfort_violations_count}")
    
    # Calculate metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_energy": np.mean(episode_energy_consumption),
        "mean_comfort_violations": np.mean(episode_comfort_violations),
        "total_episodes": n_episodes
    }
    
    print(f"\n📈 Evaluation Results:")
    print(f"   - Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"   - Mean episode length: {metrics['mean_length']:.1f}")
    print(f"   - Mean energy consumption: {metrics['mean_energy']:.2f}")
    print(f"   - Mean comfort violations: {metrics['mean_comfort_violations']:.1f}")
    
    return metrics

def run_complete_training():
    """
    Run complete PPO training pipeline for 5Zone environment.
    """
    
    print("🏢 5Zone PPO Training Pipeline")
    print("=" * 50)
    
    # Create configuration
    config = PPOTrainingConfig()
    
    # Create environments
    print("\n🔧 Creating environments...")
    train_env = create_5zone_environment(
        weather_file=config.weather_file,
        config_params=config.config_params,
        is_eval=False
    )
    
    eval_env = create_5zone_environment(
        weather_file=config.weather_file,
        config_params=config.config_params,
        is_eval=True
    )
    
    # Train the agent
    print("\n🎯 Training PPO agent...")
    trained_model = train_ppo_agent(config, train_env, eval_env)
    
    # Evaluate the trained model
    print("\n📊 Evaluating trained model...")
    evaluation_metrics = evaluate_trained_model(trained_model, eval_env, n_episodes=5)
    
    # Save evaluation results
    results_file = f"./logs/{config.experiment_name}/evaluation_results.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("5Zone PPO Training Results\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Experiment: {config.experiment_name}\n")
        f.write(f"Training timesteps: {config.total_timesteps}\n")
        f.write(f"Weather file: {config.weather_file}\n\n")
        
        f.write("Evaluation Metrics:\n")
        for key, value in evaluation_metrics.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\n💾 Evaluation results saved to: {results_file}")
    print(f"\n✅ Training completed successfully!")
    print(f"   - Model saved to: ./models/{config.experiment_name}/")
    print(f"   - Logs saved to: ./logs/{config.experiment_name}/")
    print(f"   - TensorBoard logs: ./tensorboard_logs/{config.experiment_name}/")
    
    return trained_model, evaluation_metrics

if __name__ == "__main__":
    # Run the complete training pipeline
    trained_model, metrics = run_complete_training()