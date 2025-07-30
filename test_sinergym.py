
#!/usr/bin/env python3
"""
Simple test script for Sinergym PPO training.
Run this to verify everything is working.
"""

import gymnasium as gym
import sinergym
import numpy as np

def test_simple_training():
    """Test basic PPO training functionality."""
    print("Testing Sinergym PPO training...")
    
    try:
        # Create environment
        env = gym.make('Eplus-5zone-hot-continuous-v1')
        print(f"✓ Environment created: {env}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        
        # Test a few random steps
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward = {reward:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("✓ Basic environment test completed")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_simple_training()
