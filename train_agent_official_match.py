#!/usr/bin/env python3
"""
PPO Training Script - EXACT Official Sinergym Match
==================================================

This script uses EXACTLY the same parameters as the official Sinergym PPO configuration.
Matches the official train_agent_PPO.yaml configuration file.

Usage:
    python train_agent_official_match.py
"""

import os
import numpy as np
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

import sinergym
from sinergym.utils.callbacks import LoggerEvalCallback
from sinergym.utils.wrappers import (
    NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger
)

def main():
    """Main training function - EXACT official Sinergym match"""
    
    # Configuration - EXACTLY as in official train_agent_PPO.yaml
    config = {
        'environment': 'Eplus-5zone-hot-continuous-stochastic-v1',  # OFFICIAL: stochastic version
        'episodes': 5,  # OFFICIAL: 5 episodes
        'algorithm': {
            'name': 'PPO',
            'parameters': {
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,  # OFFICIAL: 64 (not 128)
                'n_epochs': 10,
                'gamma': 0.99,  # OFFICIAL: 0.99 (not 0.9)
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'clip_range_vf': None,
                'normalize_advantage': True,
                'ent_coef': 0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,  # OFFICIAL: 0.5 (not 0.9)
                'use_sde': False,
                'sde_sample_freq': -1,
                'rollout_buffer_class': None,
                'rollout_buffer_kwargs': None,
                'target_kl': None,
                'stats_window_size': 100,
                'tensorboard_log': None,
                'policy_kwargs': None,  # OFFICIAL: None (not custom arch)
                'verbose': 1,
                'seed': None,
                'device': 'auto',
                '_init_setup_model': True
            },
            'log_interval': 1  # OFFICIAL: 1 (not 500)
        },
        'evaluation': {
            'eval_length': 1,  # OFFICIAL: 1 episode
            'eval_freq': 2     # OFFICIAL: every 2 episodes
        }
    }
    
    try:
        # Create experiment name
        experiment_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
        experiment_name = f"PPO-{config['environment']}-episodes-{config['episodes']}_{experiment_date}"
        
        # Environment parameters - EXACTLY as in official 5ZoneAutoDXVAV.yaml
        # Note: The reward parameters are already set in the environment configuration
        # We don't need to override them for the official setup
        env_params = {
            'env_name': experiment_name
        }
        
        print(f"🚀 Starting OFFICIAL Sinergym experiment: {experiment_name}")
        print(f"📦 Environment: {config['environment']} (stochastic)")
        print(f"⏱️  Episodes: {config['episodes']}")
        
        # Create training environment
        print(f"📦 Creating training environment...")
        env = gym.make(config['environment'])
        
        # Apply wrappers - EXACTLY as in official config
        print("🔧 Applying wrappers (official order)...")
        env = NormalizeAction(env=env)  # OFFICIAL: First
        env = NormalizeObservation(env=env)  # OFFICIAL: Second
        env = LoggerWrapper(env=env)  # OFFICIAL: Third
        env = CSVLogger(env=env)  # OFFICIAL: Fourth
        # OFFICIAL: WandBLogger would be here but we skip for local runs
        
        # Create evaluation environment
        print("📦 Creating evaluation environment...")
        eval_env = gym.make(config['environment'])
        
        # Apply same wrappers to eval env
        eval_env = NormalizeAction(env=eval_env)
        eval_env = NormalizeObservation(env=eval_env)
        eval_env = LoggerWrapper(env=eval_env)
        eval_env = CSVLogger(env=eval_env)
        
        # Create PPO model with OFFICIAL parameters
        print("🤖 Creating PPO model with OFFICIAL parameters...")
        model = PPO(env=env, **config['algorithm']['parameters'])
        print("✅ PPO model created with official configuration")
        
        # Calculate timesteps
        timesteps = config['episodes'] * env.get_wrapper_attr('timestep_per_episode')
        print(f"⏱️  Training for {timesteps:,} timesteps ({config['episodes']} episodes)")
        
        # Set up callbacks
        callbacks = []
        
        # Evaluation callback with OFFICIAL settings
        print("📊 Setting up evaluation callback (official settings)...")
        eval_callback = LoggerEvalCallback(
            eval_env=eval_env,
            train_env=env,
            n_eval_episodes=config['evaluation']['eval_length'],  # OFFICIAL: 1 episode
            eval_freq_episodes=config['evaluation']['eval_freq'],  # OFFICIAL: every 2 episodes
            deterministic=True
        )
        callbacks.append(eval_callback)
        
        callback = CallbackList(callbacks)
        
        # Start training with OFFICIAL log interval
        print("\n🎯 Starting training with OFFICIAL configuration...")
        print("=" * 60)
        print("📋 OFFICIAL PARAMETERS:")
        print(f"   • Environment: {config['environment']}")
        print(f"   • Episodes: {config['episodes']}")
        print(f"   • Batch size: {config['algorithm']['parameters']['batch_size']}")
        print(f"   • Gamma: {config['algorithm']['parameters']['gamma']}")
        print(f"   • Max grad norm: {config['algorithm']['parameters']['max_grad_norm']}")
        print(f"   • Log interval: {config['algorithm']['log_interval']}")
        print(f"   • Eval frequency: {config['evaluation']['eval_freq']} episodes")
        print(f"   • Eval length: {config['evaluation']['eval_length']} episode")
        print("=" * 60)
        
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=config['algorithm']['log_interval']  # OFFICIAL: 1
        )
        
        print("\n✅ Training completed with OFFICIAL configuration!")
        
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
        
        print(f"🎉 OFFICIAL experiment completed successfully!")
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