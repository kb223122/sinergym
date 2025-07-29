#!/usr/bin/env python3
"""
Sinergym Setup Script for PPO Training
======================================

This script helps set up Sinergym for PPO training by:
1. Checking system requirements
2. Installing dependencies
3. Setting up the environment
4. Testing the installation

Run this script to get started with PPO training in Sinergym.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print a step with description."""
    print(f"\n{step}. {description}")
    print("-" * 40)

def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description}")
            return True
        else:
            print(f"✗ {description}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {description}")
        print(f"Exception: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print_step(1, "Checking Python version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("✓ Python version is compatible (3.10+)")
        return True
    else:
        print("✗ Python version should be 3.10 or higher")
        print("Please upgrade Python and try again.")
        return False

def check_system_requirements():
    """Check system requirements."""
    print_step(2, "Checking system requirements")
    
    # Check OS
    system = platform.system()
    print(f"Operating System: {system}")
    
    if system == "Linux":
        print("✓ Linux is supported")
    elif system == "Darwin":
        print("✓ macOS is supported")
    elif system == "Windows":
        print("⚠ Windows support is limited, consider using WSL or Docker")
    else:
        print("✗ Unknown operating system")
        return False
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"Available memory: {memory_gb:.1f} GB")
        
        if memory_gb >= 4:
            print("✓ Sufficient memory available")
        else:
            print("⚠ Low memory detected. Training may be slow.")
    except ImportError:
        print("⚠ Could not check memory (psutil not installed)")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print_step(3, "Installing dependencies")
    
    # Core dependencies
    dependencies = [
        "gymnasium>=1.0.0",
        "numpy>=2.2.0", 
        "pandas>=2.2.2",
        "pyyaml>=6.0.2",
        "tqdm>=4.66.5",
        "xlsxwriter>=3.2.0",
        "stable-baselines3>=2.0.0",
        "matplotlib>=3.0.0"
    ]
    
    for dep in dependencies:
        success = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"Failed to install {dep}")
            return False
    
    print("✓ All dependencies installed successfully")
    return True

def setup_sinergym():
    """Set up Sinergym from the current directory."""
    print_step(4, "Setting up Sinergym")
    
    # Check if we're in the Sinergym directory
    if not os.path.exists("sinergym") and not os.path.exists("pyproject.toml"):
        print("✗ Not in Sinergym directory")
        print("Please run this script from the Sinergym repository root")
        return False
    
    # Install Sinergym in development mode
    success = run_command("pip install -e .", "Installing Sinergym in development mode")
    if not success:
        return False
    
    print("✓ Sinergym installed successfully")
    return True

def test_imports():
    """Test if all imports work correctly."""
    print_step(5, "Testing imports")
    
    imports_to_test = [
        ("gymnasium", "Gymnasium"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("sinergym", "Sinergym")
    ]
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {name}: {e}")
            return False
    
    return True

def test_environment_creation():
    """Test if we can create a Sinergym environment."""
    print_step(6, "Testing environment creation")
    
    try:
        import gymnasium as gym
        import sinergym
        
        # Check available environments
        envs = [env for env in gym.envs.registration.registry.keys() if 'Eplus' in env]
        print(f"Found {len(envs)} Sinergym environments")
        
        if len(envs) > 0:
            print("✓ Environments registered successfully")
            print(f"Available environments: {envs[:5]}...")  # Show first 5
            return True
        else:
            print("✗ No Sinergym environments found")
            return False
            
    except Exception as e:
        print(f"✗ Error testing environment creation: {e}")
        return False

def check_energyplus():
    """Check if EnergyPlus is available."""
    print_step(7, "Checking EnergyPlus")
    
    # Check environment variables
    energyplus_dir = os.environ.get('ENERGYPLUS_INSTALLATION_DIR')
    if energyplus_dir:
        print(f"EnergyPlus directory: {energyplus_dir}")
        if os.path.exists(energyplus_dir):
            print("✓ EnergyPlus installation found")
            return True
        else:
            print("✗ EnergyPlus directory does not exist")
    else:
        print("⚠ ENERGYPLUS_INSTALLATION_DIR not set")
    
    # Try to import pyenergyplus
    try:
        import pyenergyplus
        print("✓ pyenergyplus module available")
        return True
    except ImportError:
        print("✗ pyenergyplus module not available")
        print("\nEnergyPlus is required for building simulation.")
        print("You have two options:")
        print("1. Install EnergyPlus manually:")
        print("   - Download from: https://energyplus.net/downloads")
        print("   - Extract to /usr/local/EnergyPlus-24.1.0")
        print("   - Set ENERGYPLUS_INSTALLATION_DIR environment variable")
        print("\n2. Use Docker (recommended):")
        print("   docker build -t sinergym:latest --build-arg SINERGYM_EXTRAS=\"drl\" .")
        print("   docker run -it --rm sinergym:latest")
        
        return False

def create_test_script():
    """Create a simple test script."""
    print_step(8, "Creating test script")
    
    test_script = """
#!/usr/bin/env python3
\"\"\"
Simple test script for Sinergym PPO training.
Run this to verify everything is working.
\"\"\"

import gymnasium as gym
import sinergym
import numpy as np

def test_simple_training():
    \"\"\"Test basic PPO training functionality.\"\"\"
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
"""
    
    with open("test_sinergym.py", "w") as f:
        f.write(test_script)
    
    print("✓ Test script created: test_sinergym.py")
    print("Run 'python test_sinergym.py' to test the installation")
    return True

def main():
    """Main setup function."""
    print_header("SINERGYM PPO TRAINING SETUP")
    
    print("This script will help you set up Sinergym for PPO training.")
    print("It will check requirements, install dependencies, and test the installation.")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check system requirements
    if not check_system_requirements():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup Sinergym
    if not setup_sinergym():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Test environment creation
    if not test_environment_creation():
        return False
    
    # Check EnergyPlus
    energyplus_ok = check_energyplus()
    
    # Create test script
    create_test_script()
    
    # Final summary
    print_header("SETUP COMPLETED")
    
    if energyplus_ok:
        print("🎉 Setup completed successfully!")
        print("\nYou can now:")
        print("1. Run the beginner tutorial: python ppo_beginner_tutorial.py")
        print("2. Test the installation: python test_sinergym.py")
        print("3. Start training your own PPO agent")
    else:
        print("⚠ Setup completed with warnings")
        print("\nSinergym is installed but EnergyPlus is not available.")
        print("You can:")
        print("1. Install EnergyPlus manually (see instructions above)")
        print("2. Use Docker: docker build -t sinergym:latest .")
        print("3. Test basic functionality: python test_sinergym.py")
    
    print("\nNext steps:")
    print("- Read SINERGYM_PPO_ANALYSIS.md for detailed information")
    print("- Start with the beginner tutorial")
    print("- Experiment with different environments and hyperparameters")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)