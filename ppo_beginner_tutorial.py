#!/usr/bin/env python3
"""
PPO Training Tutorial for Absolute Beginners
===========================================

This is a simple, step-by-step guide to train and evaluate a PPO agent
for the Sinergym 5Zone environment. Everything is explained in detail.

What is Deep Reinforcement Learning (Deep RL)?
- It's AI that learns by trial and error
- The AI tries different actions and gets rewards
- Over time, it learns which actions get better rewards

What is PPO?
- PPO = Proximal Policy Optimization
- It's one of the best Deep RL algorithms
- It learns to make good decisions step by step

What will our AI learn?
- To control heating and cooling setpoints in a building
- To balance energy efficiency and occupant comfort
- To adapt to different weather conditions
"""

# =============================================================================
# STEP 1: IMPORTS - Get all the tools we need
# =============================================================================

print("Step 1: Importing libraries...")

import gymnasium as gym              # Standard RL environment interface
import numpy as np                   # For working with numbers and arrays
import sinergym                      # Building simulation environment
from stable_baselines3 import PPO    # The PPO algorithm
from stable_baselines3.common.callbacks import EvalCallback  # For testing during training

# Wrappers to help training work better
from sinergym.utils.wrappers import (
    NormalizeObservation,   # Makes observations easier for AI to understand
    NormalizeAction,        # Makes actions easier for AI to understand
    LoggerWrapper,          # Records what happens
    CSVLogger              # Saves data to files
)

print("✓ All libraries imported successfully!")

# =============================================================================
# STEP 2: SIMPLE CONFIGURATION - Set up our training parameters
# =============================================================================

print("\nStep 2: Setting up configuration...")

# Environment settings
ENV_NAME = 'Eplus-5zone-hot-continuous-v1'  # Which building simulation to use
EXPERIMENT_NAME = "PPO_Tutorial_Simple"      # Name for our experiment

# Training settings - these control how the AI learns
TOTAL_TIMESTEPS = 50000    # How many steps to train (start small for tutorial)
EVAL_FREQ = 10000          # How often to test the AI during training
N_EVAL_EPISODES = 2        # How many test episodes to run each time

# PPO settings - these are good default values for beginners
LEARNING_RATE = 0.0003     # How fast the AI learns (0.0003 is a good start)
N_STEPS = 2048             # Steps to collect before each AI update
BATCH_SIZE = 64            # Size of data chunks for training

print(f"✓ Configuration set:")
print(f"  - Environment: {ENV_NAME}")
print(f"  - Training steps: {TOTAL_TIMESTEPS:,}")
print(f"  - Learning rate: {LEARNING_RATE}")

# =============================================================================
# STEP 3: CREATE ENVIRONMENT - Set up the building simulation
# =============================================================================

def create_simple_environment(env_name, experiment_name, for_training=True):
    """
    Create a Sinergym environment with helpful wrappers.
    
    Args:
        env_name: Name of the environment (e.g., 'Eplus-5zone-hot-continuous-v1')
        experiment_name: Name for this experiment
        for_training: Whether this is for training (True) or evaluation (False)
    
    Returns:
        Wrapped environment ready for use
    """
    print(f"\nStep 3: Creating environment...")
    
    # Create the basic environment
    suffix = "_TRAIN" if for_training else "_EVAL"
    env = gym.make(env_name, env_name=f"{experiment_name}{suffix}")
    
    print(f"✓ Base environment created:")
    print(f"  - Name: {env.name}")
    print(f"  - Observations: {env.observation_space.shape[0]} variables")
    print(f"  - Actions: {env.action_space.shape[0]} variables (heating & cooling setpoints)")
    print(f"  - Episode length: {env.timestep_per_episode} timesteps")
    
    # Add wrappers to help training
    print("  Adding helpful wrappers...")
    
    # Normalize observations: converts sensor readings to standard scale
    # This helps because AI works better when all numbers are in similar ranges
    env = NormalizeObservation(env)
    print("    ✓ NormalizeObservation: scales sensor readings")
    
    # Normalize actions: converts actions to range [-1, 1]
    # PPO works best when actions are in this standard range
    env = NormalizeAction(env)
    print("    ✓ NormalizeAction: scales actions to [-1, 1]")
    
    # Add logging to track what happens
    env = LoggerWrapper(env)
    print("    ✓ LoggerWrapper: records interactions")
    
    env = CSVLogger(env)
    print("    ✓ CSVLogger: saves data to CSV files")
    
    print(f"✓ Environment ready for {'training' if for_training else 'evaluation'}!")
    return env

# =============================================================================
# STEP 4: CREATE PPO MODEL - Set up the AI brain
# =============================================================================

def create_simple_ppo_model(env):
    """
    Create a PPO model with simple, good settings for beginners.
    
    Args:
        env: The environment the AI will learn in
        
    Returns:
        PPO model ready for training
    """
    print(f"\nStep 4: Creating PPO model...")
    
    # Create the PPO model
    model = PPO(
        policy='MlpPolicy',         # Use a neural network policy
        env=env,                    # Environment to learn in
        learning_rate=LEARNING_RATE, # How fast to learn
        n_steps=N_STEPS,           # Steps before each update
        batch_size=BATCH_SIZE,     # Training batch size
        verbose=1                  # Print progress
    )
    
    print(f"✓ PPO model created:")
    print(f"  - Policy: Neural network (MlpPolicy)")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Steps per update: {N_STEPS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Device: {model.device}")
    
    return model

# =============================================================================
# STEP 5: TRAIN THE MODEL - Teach the AI through trial and error
# =============================================================================

def train_simple_ppo():
    """
    Train a PPO model with step-by-step explanations.
    
    Returns:
        Trained PPO model
    """
    print(f"\n" + "="*60)
    print("STARTING PPO TRAINING")
    print("="*60)
    
    # Create training environment
    train_env = create_simple_environment(ENV_NAME, EXPERIMENT_NAME, for_training=True)
    
    # Create evaluation environment (for testing during training)
    eval_env = create_simple_environment(ENV_NAME, EXPERIMENT_NAME, for_training=False)
    
    # Create the PPO model
    model = create_simple_ppo_model(train_env)
    
    # Set up evaluation callback (tests the AI periodically during training)
    print(f"\nStep 5: Setting up training monitoring...")
    eval_callback = EvalCallback(
        eval_env=eval_env,                    # Environment for testing
        best_model_save_path="./best_model/", # Where to save the best model
        log_path="./logs/",                   # Where to save logs
        eval_freq=EVAL_FREQ,                  # Test every 10,000 steps
        n_eval_episodes=N_EVAL_EPISODES,     # Run 2 test episodes each time
        deterministic=True,                   # Use best actions (no randomness) for testing
        verbose=1                             # Print test results
    )
    print(f"✓ Evaluation callback set up:")
    print(f"  - Will test every {EVAL_FREQ} steps")
    print(f"  - Will run {N_EVAL_EPISODES} test episodes each time")
    print(f"  - Will save best model to ./best_model/")
    
    # Start training!
    print(f"\nStep 6: Starting training...")
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("This will take several minutes. You can watch the progress below.")
    print()
    print("What you'll see during training:")
    print("- 'rollout/ep_rew_mean': Average reward per episode (higher is better)")
    print("- 'train/learning_rate': How fast the AI is learning")
    print("- 'eval/mean_reward': Test performance (how well AI is doing)")
    print()
    
    # This is where the actual learning happens!
    # The AI will:
    # 1. Observe the building state (temperature, weather, etc.)
    # 2. Choose actions (heating and cooling setpoints)
    # 3. Get rewards based on energy use and comfort
    # 4. Learn from this experience to make better choices next time
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True    # Show a progress bar
    )
    
    # Save the final trained model
    print(f"\nStep 7: Saving trained model...")
    model.save("./trained_model/final_model")
    print(f"✓ Model saved to: ./trained_model/final_model")
    
    # Clean up
    eval_env.close()
    train_env.close()
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    return model

# =============================================================================
# STEP 6: EVALUATE THE MODEL - Test how well our AI learned
# =============================================================================

def evaluate_simple_model(model, num_episodes=3):
    """
    Test the trained model to see how well it performs.
    
    Args:
        model: Trained PPO model
        num_episodes: Number of test episodes to run
        
    Returns:
        Dictionary with test results
    """
    print(f"\n" + "="*60)
    print("EVALUATING TRAINED MODEL")
    print("="*60)
    
    # Create evaluation environment
    eval_env = create_simple_environment(ENV_NAME, EXPERIMENT_NAME, for_training=False)
    
    print(f"Running {num_episodes} test episodes...")
    
    # Track results
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    
    for episode in range(num_episodes):
        print(f"\nTest Episode {episode + 1}/{num_episodes}")
        print("-" * 30)
        
        # Reset environment for new episode
        obs, info = eval_env.reset()
        
        # Track metrics for this episode
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        step_count = 0
        
        # Run one complete episode
        terminated = truncated = False
        while not (terminated or truncated):
            # Get action from trained model
            # deterministic=True means use the best action (no exploration)
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Track what happened
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            step_count += 1
            
            # Print progress every 1000 steps
            if step_count % 1000 == 0:
                print(f"  Step {step_count}: Current reward = {reward:.2f}")
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violation)
        
        # Print episode summary
        print(f"Episode {episode + 1} completed:")
        print(f"  - Total reward: {episode_reward:.2f}")
        print(f"  - Total energy: {episode_energy:.2f} kWh")
        print(f"  - Comfort violations: {episode_comfort_violation:.2f} °C⋅hours")
        print(f"  - Steps: {step_count}")
    
    # Calculate summary statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'mean_comfort_violation': np.mean(episode_comfort_violations),
        'episode_rewards': episode_rewards,
        'episode_energies': episode_energies,
        'episode_comfort_violations': episode_comfort_violations
    }
    
    # Print summary
    print(f"\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Episodes tested: {num_episodes}")
    print(f"Average reward: {results['mean_reward']:.2f}")
    print(f"Average energy: {results['mean_energy']:.2f} kWh")
    print(f"Average comfort violations: {results['mean_comfort_violation']:.2f} °C⋅hours")
    
    eval_env.close()
    return results

# =============================================================================
# STEP 7: COMPARE WITH RANDOM ACTIONS - See how much our AI improved
# =============================================================================

def compare_with_random(num_episodes=3):
    """
    Test random actions to see how much better our trained AI is.
    
    Args:
        num_episodes: Number of episodes to test
        
    Returns:
        Results from random policy
    """
    print(f"\n" + "="*60)
    print("COMPARING WITH RANDOM ACTIONS")
    print("="*60)
    
    # Create evaluation environment
    eval_env = create_simple_environment(ENV_NAME, EXPERIMENT_NAME, for_training=False)
    
    print(f"Running {num_episodes} episodes with random actions...")
    print("(This shows how bad completely random control would be)")
    
    # Track results
    episode_rewards = []
    episode_energies = []
    episode_comfort_violations = []
    
    for episode in range(num_episodes):
        print(f"\nRandom Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, info = eval_env.reset()
        
        # Track metrics
        episode_reward = 0
        episode_energy = 0
        episode_comfort_violation = 0
        step_count = 0
        
        # Run episode with random actions
        terminated = truncated = False
        while not (terminated or truncated):
            # Take completely random action (no intelligence!)
            action = eval_env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Track what happened
            episode_reward += reward
            episode_energy += info.get('total_power_demand', 0)
            episode_comfort_violation += info.get('total_temperature_violation', 0)
            step_count += 1
        
        # Store results
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_comfort_violations.append(episode_comfort_violation)
        
        print(f"  Random episode reward: {episode_reward:.2f}")
    
    # Calculate summary
    results = {
        'mean_reward': np.mean(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'mean_comfort_violation': np.mean(episode_comfort_violations)
    }
    
    print(f"\nRandom Policy Summary:")
    print(f"Average reward: {results['mean_reward']:.2f}")
    print(f"Average energy: {results['mean_energy']:.2f} kWh")
    print(f"Average comfort violations: {results['mean_comfort_violation']:.2f} °C⋅hours")
    
    eval_env.close()
    return results

# =============================================================================
# STEP 8: LOAD AND TEST SAVED MODEL - Use a previously trained model
# =============================================================================

def load_and_test_model(model_path):
    """
    Load a previously saved model and test it.
    
    Args:
        model_path: Path to the saved model (without .zip extension)
        
    Returns:
        Loaded model and test results
    """
    print(f"\n" + "="*60)
    print("LOADING SAVED MODEL")
    print("="*60)
    
    print(f"Loading model from: {model_path}")
    
    # Load the model
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure the model file exists and the path is correct.")
        return None, None
    
    # Test the loaded model
    print("Testing loaded model...")
    results = evaluate_simple_model(model, num_episodes=2)
    
    return model, results

# =============================================================================
# MAIN TUTORIAL FUNCTION - Put it all together
# =============================================================================

def run_complete_tutorial():
    """
    Run the complete PPO tutorial from start to finish.
    
    This function will:
    1. Train a PPO model
    2. Evaluate the trained model
    3. Compare with random actions
    4. Show the results
    """
    print("="*80)
    print("PPO TUTORIAL FOR BEGINNERS")
    print("="*80)
    print()
    print("This tutorial will teach you how to:")
    print("1. Train an AI to control building HVAC systems")
    print("2. Test how well the AI learned")
    print("3. Compare the AI with random actions")
    print()
    print("The AI will learn to balance:")
    print("- Energy efficiency (using less electricity)")
    print("- Occupant comfort (keeping good temperatures)")
    print()
    input("Press Enter to start...")
    
    try:
        # Step 1: Train the model
        print("\n🎯 PHASE 1: TRAINING THE AI")
        trained_model = train_simple_ppo()
        
        # Step 2: Evaluate the trained model
        print("\n🧪 PHASE 2: TESTING THE TRAINED AI")
        trained_results = evaluate_simple_model(trained_model, num_episodes=3)
        
        # Step 3: Compare with random actions
        print("\n🎲 PHASE 3: COMPARING WITH RANDOM ACTIONS")
        random_results = compare_with_random(num_episodes=3)
        
        # Step 4: Show final comparison
        print("\n📊 FINAL RESULTS COMPARISON")
        print("="*60)
        print(f"{'Metric':<25} {'Trained AI':<15} {'Random':<15} {'AI Better?'}")
        print("-" * 60)
        
        # Compare rewards (higher is better)
        reward_better = "✓ YES" if trained_results['mean_reward'] > random_results['mean_reward'] else "✗ NO"
        print(f"{'Average Reward':<25} {trained_results['mean_reward']:<15.1f} {random_results['mean_reward']:<15.1f} {reward_better}")
        
        # Compare energy (lower is better)
        energy_better = "✓ YES" if trained_results['mean_energy'] < random_results['mean_energy'] else "✗ NO"
        print(f"{'Energy Use (kWh)':<25} {trained_results['mean_energy']:<15.1f} {random_results['mean_energy']:<15.1f} {energy_better}")
        
        # Compare comfort (lower is better)
        comfort_better = "✓ YES" if trained_results['mean_comfort_violation'] < random_results['mean_comfort_violation'] else "✗ NO"
        print(f"{'Comfort Violations':<25} {trained_results['mean_comfort_violation']:<15.1f} {random_results['mean_comfort_violation']:<15.1f} {comfort_better}")
        
        print("\n" + "="*80)
        print("🎉 TUTORIAL COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nWhat you learned:")
        print("✓ How to create and configure a Sinergym environment")
        print("✓ How to create and train a PPO model")
        print("✓ How to evaluate a trained model")
        print("✓ How to compare AI performance with random actions")
        print("✓ How to save and load trained models")
        
        print(f"\nFiles created:")
        print(f"✓ Trained model: ./trained_model/final_model.zip")
        print(f"✓ Best model: ./best_model/best_model.zip")
        print(f"✓ Training logs: ./logs/")
        
        print(f"\nNext steps to improve your AI:")
        print("• Increase TOTAL_TIMESTEPS for longer training")
        print("• Adjust LEARNING_RATE (try 0.0001 or 0.001)")
        print("• Try different reward function weights")
        print("• Experiment with different environments")
        
        return trained_model, trained_results, random_results
        
    except Exception as e:
        print(f"\n❌ Error during tutorial: {e}")
        print("Don't worry! This is normal when learning. Try:")
        print("1. Check that all packages are installed correctly")
        print("2. Reduce TOTAL_TIMESTEPS if your computer is slow")
        print("3. Make sure you have enough disk space")
        raise

def quick_demo():
    """Run a very quick demo with minimal training (for testing)."""
    print("🚀 QUICK DEMO MODE")
    print("Training for just 5,000 steps (very fast, minimal learning)")
    
    global TOTAL_TIMESTEPS, EVAL_FREQ
    TOTAL_TIMESTEPS = 5000
    EVAL_FREQ = 2500
    
    return run_complete_tutorial()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Welcome to the PPO Tutorial for Beginners! 🤖")
    print()
    print("Choose your mode:")
    print("1. Complete tutorial (recommended) - trains for 50,000 steps")
    print("2. Quick demo - trains for only 5,000 steps (faster but less learning)")
    print("3. Load and test existing model")
    print()
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n🎓 Starting complete tutorial...")
        run_complete_tutorial()
        
    elif choice == "2":
        print("\n⚡ Starting quick demo...")
        quick_demo()
        
    elif choice == "3":
        model_path = input("\nEnter path to saved model (without .zip): ").strip()
        load_and_test_model(model_path)
        
    else:
        print("\n❓ Invalid choice. Starting complete tutorial...")
        run_complete_tutorial()
    
    print(f"\n🎯 Thanks for completing the PPO tutorial!")
    print(f"You now know the basics of training AI for building control!")