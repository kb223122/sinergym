#!/usr/bin/env python3
"""
Test Script for 5Zone PPO Training
==================================

This script tests the ppo_5zone_training.py implementation to ensure it works correctly.
It runs a minimal training session to verify all components function properly.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Import the training functions
from ppo_5zone_training import (
    PPOTrainingConfig,
    create_5zone_environment,
    create_ppo_model,
    setup_training_callbacks,
    train_ppo_agent,
    evaluate_trained_model
)

def test_environment_creation():
    """Test that 5Zone environments can be created successfully."""
    print("🧪 Testing 5Zone environment creation...")
    
    config = PPOTrainingConfig()
    
    try:
        # Create training environment
        train_env = create_5zone_environment(
            weather_file=config.weather_file,
            config_params=config.config_params,
            is_eval=False
        )
        print("✅ Training environment created successfully")
        
        # Create evaluation environment
        eval_env = create_5zone_environment(
            weather_file=config.weather_file,
            config_params=config.config_params,
            is_eval=True
        )
        print("✅ Evaluation environment created successfully")
        
        return train_env, eval_env
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return None, None

def test_ppo_model_creation(train_env):
    """Test that PPO model can be created successfully."""
    print("🧪 Testing PPO model creation...")
    
    config = PPOTrainingConfig()
    
    try:
        model = create_ppo_model(train_env, config)
        print("✅ PPO model created successfully")
        return model
        
    except Exception as e:
        print(f"❌ PPO model creation failed: {e}")
        return None

def test_training_callbacks(config, eval_env):
    """Test that training callbacks can be created successfully."""
    print("🧪 Testing training callbacks creation...")
    
    try:
        callbacks = setup_training_callbacks(config, eval_env)
        print(f"✅ Training callbacks created successfully ({len(callbacks)} callbacks)")
        return callbacks
        
    except Exception as e:
        print(f"❌ Training callbacks creation failed: {e}")
        return None

def test_minimal_training():
    """Test a minimal training session."""
    print("🧪 Testing minimal training session...")
    
    config = PPOTrainingConfig()
    
    # Reduce training parameters for quick test
    config.total_timesteps = 1000  # Very short training
    config.eval_freq = 500
    config.save_freq = 1000
    config.n_eval_episodes = 2
    
    try:
        # Create environments
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
        trained_model = train_ppo_agent(config, train_env, eval_env)
        
        print("✅ Minimal training completed successfully")
        return trained_model, eval_env
        
    except Exception as e:
        print(f"❌ Minimal training failed: {e}")
        return None, None

def test_evaluation(trained_model, eval_env):
    """Test model evaluation."""
    print("🧪 Testing model evaluation...")
    
    try:
        metrics = evaluate_trained_model(trained_model, eval_env, n_episodes=2)
        print("✅ Model evaluation completed successfully")
        return metrics
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return None

def run_all_tests():
    """Run all tests."""
    print("🚀 Running 5Zone PPO Training Tests")
    print("=" * 50)
    
    # Test 1: Environment creation
    train_env, eval_env = test_environment_creation()
    if train_env is None or eval_env is None:
        print("❌ Environment creation test failed. Stopping tests.")
        return False
    
    # Test 2: PPO model creation
    model = test_ppo_model_creation(train_env)
    if model is None:
        print("❌ PPO model creation test failed. Stopping tests.")
        return False
    
    # Test 3: Training callbacks
    config = PPOTrainingConfig()
    callbacks = test_training_callbacks(config, eval_env)
    if callbacks is None:
        print("❌ Training callbacks test failed. Stopping tests.")
        return False
    
    # Test 4: Minimal training
    trained_model, eval_env = test_minimal_training()
    if trained_model is None:
        print("❌ Minimal training test failed. Stopping tests.")
        return False
    
    # Test 5: Model evaluation
    metrics = test_evaluation(trained_model, eval_env)
    if metrics is None:
        print("❌ Model evaluation test failed.")
        return False
    
    print("\n🎉 All tests passed successfully!")
    print("✅ 5Zone PPO training code is working correctly")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ Ready to run full training with: python ppo_5zone_training.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")