#!/usr/bin/env python3
"""
Simple PPO Training Script for Sinergym 3.7.3
=============================================

A simplified version that avoids problematic imports and works reliably
with Sinergym version 3.7.3 and Python 3.12.4.

Usage:
    python train_agent_simple.py
"""

import os
import numpy as np
from datetime import datetime
import yaml

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

import sinergym
from sinergym.utils.callbacks import LoggerEvalCallback
from sinergym.utils.wrappers import (
    NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger
)

def main():
    """Main training function"""
    
    # Configuration (hardcoded for simplicity)
    config = {
        'environment': 'Eplus-5zone-hot-continuous-v1',
        'episodes': 30,
        'algorithm': {
            'name': 'PPO',
            'parameters': {
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 128,
                'n_epochs': 10,
                'gamma': 0.9,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.9,
                'normalize_advantage': True,
                'verbose': 1,
                'seed': 42,
                'policy_kwargs': {
                    'net_arch': [64, 64]
                }
            },
            'log_interval': 500
        },
        'evaluation': {
            'eval_length': 3,
            'eval_freq': 5
        }
    }
    
    # Environment parameters
    env_params = {
        'timesteps_per_hour': 4,
        'runperiod': [1, 1, 1991, 31, 12, 1991],
        'reward': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': [20.0, 23.5],
            'range_comfort_summer': [23.0, 26.0],
            'summer_start': [6, 1],
            'summer_final': [9, 30],
            'energy_weight': 0.5,
            'lambda_energy': 0.0001,
            'lambda_temperature': 1.0
        }
    }
    
    try:
        # Create experiment name
        experiment_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
        experiment_name = f"PPO-{config['environment']}-episodes-{config['episodes']}_{experiment_date}"
        env_params['env_name'] = experiment_name
        
        print(f"🚀 Starting experiment: {experiment_name}")
        
        # Create training environment
        print(f"📦 Creating training environment: {config['environment']}")
        env = gym.make(config['environment'], **env_params)
        
        # Apply wrappers
        print("🔧 Applying wrappers...")
        env = NormalizeObservation(env=env)
        env = NormalizeAction(env=env)
        env = LoggerWrapper(env=env)
        env = CSVLogger(env=env)
        
        # Create evaluation environment
        print("📦 Creating evaluation environment...")
        eval_env_params = env_params.copy()
        eval_env_params['env_name'] = experiment_name + '-EVAL'
        eval_env = gym.make(config['environment'], **eval_env_params)
        
        # Apply same wrappers to eval env
        eval_env = NormalizeObservation(env=eval_env)
        eval_env = NormalizeAction(env=eval_env)
        eval_env = LoggerWrapper(env=eval_env)
        eval_env = CSVLogger(env=eval_env)
        
        # Create PPO model
        print("🤖 Creating PPO model...")
        model = PPO(env=env, **config['algorithm']['parameters'])
        print("✅ PPO model created successfully")
        
        # Calculate timesteps
        timesteps = config['episodes'] * env.get_wrapper_attr('timestep_per_episode')
        print(f"⏱️  Training for {timesteps:,} timesteps ({config['episodes']} episodes)")
        
        # Set up callbacks
        callbacks = []
        
        # Evaluation callback
        print("📊 Setting up evaluation callback...")
        eval_callback = LoggerEvalCallback(
            eval_env=eval_env,
            train_env=env,
            n_eval_episodes=config['evaluation']['eval_length'],
            eval_freq_episodes=config['evaluation']['eval_freq'],
            deterministic=True
        )
        callbacks.append(eval_callback)
        
        callback = CallbackList(callbacks)
        
        # Start training
        print("\n🎯 Starting training...")
        print("=" * 60)
        
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=config['algorithm']['log_interval']
        )
        
        print("\n✅ Training completed!")
        
        # Save model
        workspace_path = env.get_wrapper_attr('workspace_path')
        model_save_path = os.path.join(workspace_path, 'model')
        model.save(model_save_path)
        print(f"💾 Model saved to: {model_save_path}.zip")
        
        # Cleanup
        if env.get_wrapper_attr('is_running'):
            env.close()
        if eval_env.get_wrapper_attr('is_running'):
            eval_env.close()
        
        print(f"🎉 Experiment completed successfully!")
        print(f"📁 Results saved in: {workspace_path}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save model and close envs
        try:
            if 'model' in locals() and 'env' in locals():
                workspace_path = env.get_wrapper_attr('workspace_path')
                model_save_path = os.path.join(workspace_path, 'model_error')
                model.save(model_save_path)
                print(f"💾 Emergency model save to: {model_save_path}.zip")
        except:
            pass
        
        try:
            if 'env' in locals() and env.get_wrapper_attr('is_running'):
                env.close()
            if 'eval_env' in locals() and eval_env.get_wrapper_attr('is_running'):
                eval_env.close()
        except:
            pass
        
        raise e

if __name__ == "__main__":
    main()