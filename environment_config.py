#!/usr/bin/env python3
"""
Environment Configuration Utility for 5Zone Environment
=====================================================

Simple utility to configure reward weights and run periods for the
Eplus-5zone-hot-continuous-v1 environment.

Usage:
    from environment_config import get_config
    config = get_config('energy_focused', 'summer_only')
"""

from typing import Dict, Tuple

def get_reward_config(config_name: str = 'default') -> Dict:
    """
    Get reward configuration by name.
    
    Args:
        config_name (str): Configuration name
    
    Returns:
        dict: Reward configuration
    """
    
    configs = {
        'default': {
            'energy_weight': 0.5,
            'lambda_energy': 1e-4,
            'lambda_temperature': 1.0,
            'description': 'Balanced energy and comfort'
        },
        'energy_focused': {
            'energy_weight': 0.8,
            'lambda_energy': 2e-4,
            'lambda_temperature': 0.5,
            'description': 'Prioritizes energy savings'
        },
        'comfort_focused': {
            'energy_weight': 0.2,
            'lambda_energy': 5e-5,
            'lambda_temperature': 2.0,
            'description': 'Prioritizes occupant comfort'
        },
        'extreme_energy': {
            'energy_weight': 0.9,
            'lambda_energy': 5e-4,
            'lambda_temperature': 0.1,
            'description': 'Maximum energy savings'
        },
        'extreme_comfort': {
            'energy_weight': 0.1,
            'lambda_energy': 1e-5,
            'lambda_temperature': 5.0,
            'description': 'Maximum comfort priority'
        },
        'custom': {
            'energy_weight': 0.7,
            'lambda_energy': 2e-4,
            'lambda_temperature': 0.8,
            'description': 'Custom configuration'
        }
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown reward config '{config_name}', using default")
        config_name = 'default'
    
    return configs[config_name]

def get_run_period_config(config_name: str = 'default') -> Dict:
    """
    Get run period configuration by name.
    
    Args:
        config_name (str): Configuration name
    
    Returns:
        dict: Run period configuration
    """
    
    configs = {
        'default': {
            'timestep_per_hour': 1,
            'start_month': 1,
            'start_day': 1,
            'end_month': 12,
            'end_day': 31,
            'description': 'Full year with 1-hour timesteps'
        },
        'summer_only': {
            'timestep_per_hour': 1,
            'start_month': 6,
            'start_day': 1,
            'end_month': 8,
            'end_day': 31,
            'description': 'Summer months (June-August)'
        },
        'winter_only': {
            'timestep_per_hour': 1,
            'start_month': 12,
            'start_day': 1,
            'end_month': 2,
            'end_day': 28,
            'description': 'Winter months (December-February)'
        },
        'spring_only': {
            'timestep_per_hour': 1,
            'start_month': 3,
            'start_day': 1,
            'end_month': 5,
            'end_day': 31,
            'description': 'Spring months (March-May)'
        },
        'two_hour_timesteps': {
            'timestep_per_hour': 2,
            'start_month': 1,
            'start_day': 1,
            'end_month': 12,
            'end_day': 31,
            'description': 'Full year with 2-hour timesteps'
        },
        'four_hour_timesteps': {
            'timestep_per_hour': 4,
            'start_month': 1,
            'start_day': 1,
            'end_month': 12,
            'end_day': 31,
            'description': 'Full year with 4-hour timesteps'
        },
        'summer_two_hour': {
            'timestep_per_hour': 2,
            'start_month': 6,
            'start_day': 1,
            'end_month': 8,
            'end_day': 31,
            'description': 'Summer with 2-hour timesteps'
        }
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown run period config '{config_name}', using default")
        config_name = 'default'
    
    return configs[config_name]

def get_training_config(config_name: str = 'default') -> Dict:
    """
    Get training configuration by name.
    
    Args:
        config_name (str): Configuration name
    
    Returns:
        dict: Training configuration
    """
    
    configs = {
        'default': {
            'total_timesteps': 50000,
            'learning_rate': 3e-4,
            'eval_episodes': 3,
            'description': 'Standard training'
        },
        'quick_test': {
            'total_timesteps': 10000,
            'learning_rate': 3e-4,
            'eval_episodes': 2,
            'description': 'Quick test training'
        },
        'long_training': {
            'total_timesteps': 200000,
            'learning_rate': 3e-4,
            'eval_episodes': 5,
            'description': 'Extended training'
        },
        'high_lr': {
            'total_timesteps': 50000,
            'learning_rate': 1e-3,
            'eval_episodes': 3,
            'description': 'High learning rate'
        },
        'low_lr': {
            'total_timesteps': 50000,
            'learning_rate': 1e-4,
            'eval_episodes': 3,
            'description': 'Low learning rate'
        }
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown training config '{config_name}', using default")
        config_name = 'default'
    
    return configs[config_name]

def get_config(reward_config_name: str = 'default', 
               run_period_config_name: str = 'default',
               training_config_name: str = 'default') -> Dict:
    """
    Get complete configuration.
    
    Args:
        reward_config_name (str): Reward configuration name
        run_period_config_name (str): Run period configuration name
        training_config_name (str): Training configuration name
    
    Returns:
        dict: Complete configuration
    """
    
    config = {
        'reward': get_reward_config(reward_config_name),
        'run_period': get_run_period_config(run_period_config_name),
        'training': get_training_config(training_config_name)
    }
    
    return config

def print_config(config: Dict):
    """Print configuration details."""
    
    print("🔧 ENVIRONMENT CONFIGURATION")
    print("=" * 60)
    
    # Reward configuration
    reward = config['reward']
    print(f"\n📊 Reward Configuration: {reward['description']}")
    print(f"   Energy Weight: {reward['energy_weight']:.2f} ({reward['energy_weight']*100:.0f}%)")
    print(f"   Lambda Energy: {reward['lambda_energy']:.2e}")
    print(f"   Lambda Temperature: {reward['lambda_temperature']:.2f}")
    
    # Run period configuration
    run_period = config['run_period']
    print(f"\n⏱️ Run Period Configuration: {run_period['description']}")
    print(f"   Timesteps per Hour: {run_period['timestep_per_hour']}")
    print(f"   Run Period: ({run_period['start_month']}, {run_period['start_day']}, {run_period['end_month']}, {run_period['end_day']})")
    
    # Training configuration
    training = config['training']
    print(f"\n🎯 Training Configuration: {training['description']}")
    print(f"   Total Timesteps: {training['total_timesteps']:,}")
    print(f"   Learning Rate: {training['learning_rate']}")
    print(f"   Evaluation Episodes: {training['eval_episodes']}")

def list_available_configs():
    """List all available configurations."""
    
    print("📋 AVAILABLE CONFIGURATIONS")
    print("=" * 60)
    
    print("\n🎯 Reward Configurations:")
    reward_configs = ['default', 'energy_focused', 'comfort_focused', 'extreme_energy', 'extreme_comfort', 'custom']
    for config in reward_configs:
        cfg = get_reward_config(config)
        print(f"   {config}: {cfg['description']}")
    
    print("\n⏱️ Run Period Configurations:")
    run_period_configs = ['default', 'summer_only', 'winter_only', 'spring_only', 'two_hour_timesteps', 'four_hour_timesteps', 'summer_two_hour']
    for config in run_period_configs:
        cfg = get_run_period_config(config)
        print(f"   {config}: {cfg['description']}")
    
    print("\n🎓 Training Configurations:")
    training_configs = ['default', 'quick_test', 'long_training', 'high_lr', 'low_lr']
    for config in training_configs:
        cfg = get_training_config(config)
        print(f"   {config}: {cfg['description']}")

# Example usage
if __name__ == "__main__":
    print("🚀 Environment Configuration Utility")
    print("=" * 50)
    
    # List available configurations
    list_available_configs()
    
    # Example configurations
    print("\n" + "="*60)
    print("📋 EXAMPLE CONFIGURATIONS")
    print("=" * 60)
    
    # Energy-focused with summer-only
    config1 = get_config('energy_focused', 'summer_only', 'quick_test')
    print("\n1. Energy-focused, Summer-only, Quick test:")
    print_config(config1)
    
    # Comfort-focused with full year
    config2 = get_config('comfort_focused', 'default', 'default')
    print("\n2. Comfort-focused, Full year, Standard training:")
    print_config(config2)
    
    # Custom configuration
    config3 = get_config('custom', 'two_hour_timesteps', 'long_training')
    print("\n3. Custom, Two-hour timesteps, Long training:")
    print_config(config3)