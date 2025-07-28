#!/usr/bin/env python3
"""
Test Script for Simplified PPO Training
=======================================

This script tests the direct_ppo_training_simple.py implementation to ensure it works correctly.
It runs a minimal training session to verify all components function properly.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Import the training functions
from direct_ppo_training_simple import (
    PPOTrainingConfig,
    create_training_environment,
    create_evaluation_environment,
    create_ppo_model,
    setup_training_callbacks,
    train_ppo_agent,
    evaluate_trained_model,
    compare_with_random_policy
)

def test_environment_creation():
    """Test that environments can be created successfully."""
    print("🧪 Testing environment creation...")
    
    config = PPOTrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=10_000,  # Very small for testing
        experiment_name="Test_Env_Creation"
    )
    
    try:
        train_env = create_training_environment(config)
        eval_env = create_evaluation_environment(config)
        
        # Test basic environment properties
        assert train_env.observation_space.shape == (4,), f"Expected observation space (4,), got {train_env.observation_space.shape}"
        assert train_env.action_space.n == 2, f"Expected action space with 2 actions, got {train_env.action_space.n}"
        
        print("✅ Environment creation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

def test_model_creation():
    """Test that PPO model can be created successfully."""
    print("🤖 Testing model creation...")
    
    config = PPOTrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=10_000,
        experiment_name="Test_Model_Creation"
    )
    
    try:
        train_env = create_training_environment(config)
        model = create_ppo_model(train_env, config)
        
        # Test basic model properties
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'learn'), "Model should have learn method"
        assert hasattr(model, 'predict'), "Model should have predict method"
        
        print("✅ Model creation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_callbacks():
    """Test that callbacks can be set up successfully."""
    print("📊 Testing callbacks...")
    
    config = PPOTrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=10_000,
        experiment_name="Test_Callbacks"
    )
    
    try:
        eval_env = create_evaluation_environment(config)
        callbacks = setup_training_callbacks(config, eval_env)
        
        # Test that callbacks were created
        assert len(callbacks) > 0, "Should have at least one callback"
        
        print("✅ Callbacks: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Callback test failed: {e}")
        return False

def test_minimal_training():
    """Test minimal training session."""
    print("🎯 Testing minimal training...")
    
    config = PPOTrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=5_000,  # Very small for testing
        eval_freq=2_000,
        experiment_name="Test_Minimal_Training"
    )
    
    try:
        model, train_env, eval_env = train_ppo_agent(config)
        
        # Test that model was trained
        assert model is not None, "Model should not be None after training"
        
        # Test evaluation
        trained_metrics = evaluate_trained_model(model, eval_env, n_episodes=5)
        random_metrics = compare_with_random_policy(eval_env, n_episodes=5)
        
        # Test that trained model performs better than random
        assert trained_metrics['mean_reward'] > random_metrics['mean_reward'], \
            f"Trained model ({trained_metrics['mean_reward']:.2f}) should perform better than random ({random_metrics['mean_reward']:.2f})"
        
        print("✅ Minimal training: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Minimal training failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 RUNNING SIMPLIFIED PPO TRAINING TESTS")
    print("=" * 60)
    
    tests = [
        ("Environment Creation", test_environment_creation),
        ("Model Creation", test_model_creation),
        ("Callbacks", test_callbacks),
        ("Minimal Training", test_minimal_training),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    print(f"\n🔧 You can now run the complete training with:")
    print(f"   python direct_ppo_training_simple.py")

if __name__ == "__main__":
    main()