#!/usr/bin/env python3
"""
Test Script for Direct PPO Training
===================================

This script tests the direct_ppo_training.py implementation to ensure it works correctly.
It runs a minimal training session to verify all components function properly.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Import the training functions
from direct_ppo_training import (
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
    
    config = PPOTrainingConfig()
    
    try:
        # Test training environment creation
        train_env = create_training_environment(config)
        print("✓ Training environment created successfully")
        
        # Test evaluation environment creation
        eval_env = create_evaluation_environment(config)
        print("✓ Evaluation environment created successfully")
        
        # Test basic environment properties
        obs, info = train_env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Action space: {train_env.action_space}")
        print(f"  - Observation space: {train_env.observation_space}")
        
        # Test environment step
        action = train_env.action_space.sample()
        obs, reward, terminated, truncated, info = train_env.step(action)
        print(f"✓ Environment step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False

def test_model_creation():
    """Test that PPO model can be created successfully."""
    print("\n🤖 Testing model creation...")
    
    config = PPOTrainingConfig()
    
    try:
        # Create environment
        train_env = create_training_environment(config)
        
        # Create model
        model = create_ppo_model(train_env, config)
        print("✓ PPO model created successfully")
        print(f"  - Device: {model.device}")
        print(f"  - Policy: {model.policy}")
        
        # Test model prediction
        obs, info = train_env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"✓ Model prediction successful")
        print(f"  - Action shape: {action.shape}")
        print(f"  - Action range: [{action.min():.2f}, {action.max():.2f}]")
        
        # Clean up
        train_env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_minimal_training():
    """Test minimal training session."""
    print("\n🎯 Testing minimal training...")
    
    # Create config with minimal settings for quick test
    config = PPOTrainingConfig()
    config.total_timesteps = 1000  # Very short training
    config.eval_freq = 500
    config.n_eval_episodes = 1
    
    try:
        # Run minimal training
        model, train_env, eval_env = train_ppo_agent(config)
        print("✓ Minimal training completed successfully")
        
        # Test evaluation
        results = evaluate_trained_model(model, eval_env, config, num_episodes=1)
        print("✓ Model evaluation successful")
        print(f"  - Mean reward: {results['mean_reward']:.2f}")
        print(f"  - Mean energy: {results['mean_energy']:.2f} kWh")
        
        # Test random policy comparison
        random_results = compare_with_random_policy(eval_env, config, num_episodes=1)
        print("✓ Random policy comparison successful")
        print(f"  - Random mean reward: {random_results['mean_reward']:.2f}")
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Minimal training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration class."""
    print("\n⚙️  Testing configuration...")
    
    try:
        config = PPOTrainingConfig()
        config.print_config()
        print("✓ Configuration created and printed successfully")
        
        # Test configuration properties
        assert config.env_name == 'Eplus-5zone-hot-continuous-v1'
        assert config.learning_rate == 3e-4
        assert config.n_steps == 2048
        assert config.batch_size == 64
        print("✓ Configuration properties verified")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_callbacks():
    """Test callback setup."""
    print("\n📊 Testing callbacks...")
    
    config = PPOTrainingConfig()
    
    try:
        # Create environments
        train_env = create_training_environment(config)
        eval_env = create_evaluation_environment(config)
        
        # Test callback setup
        callbacks = setup_training_callbacks(config, eval_env)
        print("✓ Callbacks setup successful")
        print(f"  - Number of callbacks: {len(callbacks.callbacks)}")
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Callback test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("🧪 RUNNING DIRECT PPO TRAINING TESTS")
    print("="*60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Environment Creation", test_environment_creation),
        ("Model Creation", test_model_creation),
        ("Callbacks", test_callbacks),
        ("Minimal Training", test_minimal_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The direct PPO training implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print(f"\n🚀 You can now run the complete training with:")
        print(f"   python direct_ppo_training.py")
    else:
        print(f"\n🔧 Please fix the failing tests before running the complete training.")
    
    sys.exit(0 if success else 1)