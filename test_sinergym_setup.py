#!/usr/bin/env python3
"""
Test Sinergym Setup
This script verifies that Sinergym is properly installed and working.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import gymnasium as gym
        print("✓ gymnasium imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import gymnasium: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False
    
    try:
        import sinergym
        print("✓ sinergym imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sinergym: {e}")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("✓ stable_baselines3 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import stable_baselines3: {e}")
        print("  Note: This is optional for basic usage")
    
    return True

def test_environment_creation():
    """Test that environments can be created."""
    print("\nTesting environment creation...")
    
    try:
        import gymnasium as gym
        import sinergym
        
        # Test demo environment
        env = gym.make('Eplus-demo-v1')
        print("✓ Demo environment created successfully")
        print(f"  Environment name: {env.name}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print("✓ Environment reset successfully")
        print(f"  Observation shape: {obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment step successful")
        print(f"  Reward: {reward}")
        
        # Close environment
        env.close()
        print("✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_available_environments():
    """List available environments."""
    print("\nAvailable Sinergym environments:")
    
    try:
        import gymnasium as gym
        
        available_envs = [env_id for env_id in gym.envs.registration.registry.keys() 
                         if env_id.startswith('Eplus')]
        
        for i, env_id in enumerate(available_envs[:10]):  # Show first 10
            print(f"  {i+1:2d}. {env_id}")
        
        if len(available_envs) > 10:
            print(f"  ... and {len(available_envs)-10} more environments")
        
        print(f"\nTotal Sinergym environments: {len(available_envs)}")
        
    except Exception as e:
        print(f"✗ Failed to list environments: {e}")

def test_wrappers():
    """Test that wrappers can be imported and used."""
    print("\nTesting wrappers...")
    
    try:
        from sinergym.utils.wrappers import (
            NormalizeAction,
            NormalizeObservation,
            LoggerWrapper,
            CSVLogger
        )
        print("✓ Wrappers imported successfully")
        
        # Test wrapper application
        import gymnasium as gym
        import sinergym
        
        env = gym.make('Eplus-demo-v1')
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        env = LoggerWrapper(env)
        env = CSVLogger(env)
        
        print("✓ Wrappers applied successfully")
        
        # Test wrapped environment
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✓ Wrapped environment works correctly")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Wrapper test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Sinergym Setup Test ===\n")
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n✗ Import test failed. Please check your installation.")
        return
    
    # Test environment creation
    env_ok = test_environment_creation()
    
    if not env_ok:
        print("\n✗ Environment test failed. Please check your EnergyPlus installation.")
        return
    
    # Test wrappers
    wrapper_ok = test_wrappers()
    
    # List available environments
    test_available_environments()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"Environment: {'✓' if env_ok else '✗'}")
    print(f"Wrappers: {'✓' if wrapper_ok else '✗'}")
    
    if imports_ok and env_ok:
        print("\n🎉 Sinergym is ready to use!")
        print("\nNext steps:")
        print("1. Run 'python basic_example.py' for a simple demonstration")
        print("2. Run 'python rule_based_controller_example.py' for control strategies")
        print("3. Run 'python reinforcement_learning_example.py' for RL training")
        print("4. Explore the Jupyter notebooks in the 'examples/' directory")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()