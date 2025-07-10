#!/usr/bin/env python3
"""
Compatible PPO Training Script for Sinergym 3.7.3
=================================================

This script is adapted to work with Sinergym version 3.7.3 and Python 3.12.4.
It removes incompatible imports and functions that don't exist in this version.

Usage:
    python train_agent_local_compatible.py --configuration train_PPO.yaml
"""

import argparse
import sys
import traceback
from datetime import datetime
import os

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger

import sinergym
from sinergym.utils.callbacks import LoggerEvalCallback
from sinergym.utils.rewards import LinearReward
from sinergym.utils.wrappers import (
    NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger
)

# ---------------------------------------------------------------------------- #
#                             Parameters definition                            #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--configuration',
    '-conf',
    required=True,
    type=str,
    dest='configuration',
    help='Path to experiment configuration (YAML file)'
)
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                             Read yaml parameters                             #
# ---------------------------------------------------------------------------- #
with open(args.configuration, 'r') as yaml_conf:
    conf = yaml.safe_load(yaml_conf)

try:
    # ---------------------------------------------------------------------------- #
    #                               Register run name                              #
    # ---------------------------------------------------------------------------- #
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
    experiment_name = conf['algorithm']['name'] + '-' + conf['environment'] + \
        '-episodes-' + str(conf['episodes'])
    if conf.get('id'):
        experiment_name += '-id-' + str(conf['id'])
    experiment_name += '_' + experiment_date

    print(f"🚀 Starting experiment: {experiment_name}")

    # ---------------------------------------------------------------------------- #
    #                           Environment construction                           #
    # ---------------------------------------------------------------------------- #
    # Get environment parameters
    env_params = conf.get('env_params', {})
    env_params.update({'env_name': experiment_name})
    
    print(f"📦 Creating training environment: {conf['environment']}")
    env = gym.make(conf['environment'], **env_params)

    # Create evaluation environment if enabled
    eval_env = None
    if conf.get('evaluation'):
        eval_name = conf['evaluation'].get(
            'name', env.get_wrapper_attr('name') + '-EVAL')
        env_params.update({'env_name': eval_name})
        print(f"📦 Creating evaluation environment: {eval_name}")
        eval_env = gym.make(conf['environment'], **env_params)

    # ---------------------------------------------------------------------------- #
    #                                   Wrappers                                   #
    # ---------------------------------------------------------------------------- #
    print("🔧 Applying wrappers...")
    
    # Apply wrappers based on configuration
    if conf.get('wrappers'):
        for wrapper in conf['wrappers']:
            for key, parameters in wrapper.items():
                wrapper_class = eval(key)
                print(f"  - Applying {key}")
                env = wrapper_class(env=env, **parameters)
                if eval_env is not None:
                    # Don't apply WandBLogger to evaluation env
                    if key != 'WandBLogger':
                        eval_env = wrapper_class(env=eval_env, **parameters)

    # ---------------------------------------------------------------------------- #
    #                           Defining model (algorithm)                         #
    # ---------------------------------------------------------------------------- #
    alg_name = conf['algorithm']['name']
    alg_params = conf['algorithm'].get('parameters', {'policy': 'MlpPolicy'})
    
    print(f"🤖 Creating {alg_name} model...")
    
    # Create model from scratch
    if conf.get('model') is None:
        try:
            model = eval(alg_name)(env=env, **alg_params)
            print(f"✅ {alg_name} model created successfully")
        except NameError:
            raise NameError(
                f'Algorithm {alg_name} does not exist. It must be a valid SB3 algorithm.')

    # Load existing model (if specified)
    else:
        print("📂 Loading existing model...")
        model_path = conf['model'].get('local_path')
        if not model_path:
            raise ValueError("Model path not specified in configuration")
        
        try:
            model = eval(alg_name).load(model_path)
            model.set_env(env)
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model from {model_path}: {e}")

    # ---------------------------------------------------------------------------- #
    #       Calculating total training timesteps based on number of episodes       #
    # ---------------------------------------------------------------------------- #
    timesteps = conf['episodes'] * (env.get_wrapper_attr('timestep_per_episode'))
    print(f"⏱️  Training for {timesteps:,} timesteps ({conf['episodes']} episodes)")

    # ---------------------------------------------------------------------------- #
    #                                   CALLBACKS                                  #
    # ---------------------------------------------------------------------------- #
    callbacks = []

    # Set up Evaluation and best model saving
    if conf.get('evaluation') and eval_env is not None:
        print("📊 Setting up evaluation callback...")
        eval_callback = LoggerEvalCallback(
            eval_env=eval_env,
            train_env=env,
            n_eval_episodes=conf['evaluation']['eval_length'],
            eval_freq_episodes=conf['evaluation']['eval_freq'],
            deterministic=True
        )
        callbacks.append(eval_callback)
        print(f"  - Evaluation every {conf['evaluation']['eval_freq']} episodes")
        print(f"  - {conf['evaluation']['eval_length']} episodes per evaluation")

    callback = CallbackList(callbacks)

    # ---------------------------------------------------------------------------- #
    #                                   TRAINING                                   #
    # ---------------------------------------------------------------------------- #
    print("\n🎯 Starting training...")
    print("=" * 60)
    
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=conf['algorithm']['log_interval']
    )
    
    print("\n✅ Training completed!")
    
    # Save the final model
    workspace_path = env.get_wrapper_attr('workspace_path')
    model_save_path = os.path.join(workspace_path, 'model')
    model.save(model_save_path)
    print(f"💾 Model saved to: {model_save_path}.zip")

    # ---------------------------------------------------------------------------- #
    #                                   CLEANUP                                    #
    # ---------------------------------------------------------------------------- #
    # Close environments properly
    if env.get_wrapper_attr('is_running'):
        env.close()
    if eval_env and eval_env.get_wrapper_attr('is_running'):
        eval_env.close()
    
    print(f"🎉 Experiment completed successfully!")
    print(f"📁 Results saved in: {workspace_path}")

# ---------------------------------------------------------------------------- #
#                              ERROR HANDLING                                    #
# ---------------------------------------------------------------------------- #
except (Exception, KeyboardInterrupt) as err:
    print("❌ Error or interruption in process detected")
    print(traceback.format_exc())
    
    # Try to save current model state if possible
    try:
        if 'model' in locals() and 'env' in locals():
            workspace_path = env.get_wrapper_attr('workspace_path')
            model_save_path = os.path.join(workspace_path, 'model_error')
            model.save(model_save_path)
            print(f"💾 Emergency model save to: {model_save_path}.zip")
    except:
        pass
    
    # Try to close environment
    try:
        if 'env' in locals() and env.get_wrapper_attr('is_running'):
            env.close()
        if 'eval_env' in locals() and eval_env and eval_env.get_wrapper_attr('is_running'):
            eval_env.close()
    except:
        pass
    
    raise err