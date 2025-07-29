#!/usr/bin/env python3
"""
PPO Demo Without EnergyPlus
===========================

This script demonstrates PPO training concepts using a simple custom environment
that simulates building control without requiring EnergyPlus.

This is useful for:
1. Understanding PPO concepts
2. Testing the training pipeline
3. Learning before setting up EnergyPlus
4. Demonstrating the code structure

The environment simulates a simple building with:
- Temperature control
- Energy consumption
- Comfort violations
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt

# =============================================================================
# SIMPLE BUILDING ENVIRONMENT
# =============================================================================

class SimpleBuildingEnv(gym.Env):
    """
    A simple building environment for PPO training demonstration.
    
    This environment simulates:
    - Building temperature control
    - Energy consumption
    - Comfort violations
    - Weather influence
    """
    
    def __init__(self, max_steps=1000):
        super().__init__()
        
        # Environment parameters
        self.max_steps = max_steps
        self.current_step = 0
        
        # Building parameters
        self.target_temp = 22.0  # Target temperature (°C)
        self.comfort_range = (20.0, 24.0)  # Comfort range (°C)
        self.outdoor_temp = 25.0  # Outdoor temperature (°C)
        self.current_temp = 22.0  # Current indoor temperature (°C)
        
        # HVAC parameters
        self.heating_power = 0.0  # Heating power (kW)
        self.cooling_power = 0.0  # Cooling power (kW)
        self.max_hvac_power = 10.0  # Maximum HVAC power (kW)
        
        # Define action space (heating and cooling setpoints)
        # Actions are continuous values between -1 and 1
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # [heating, cooling]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space
        # [current_temp, outdoor_temp, heating_power, cooling_power, time_of_day]
        self.observation_space = spaces.Box(
            low=np.array([10.0, -10.0, 0.0, 0.0, 0.0]),
            high=np.array([35.0, 40.0, self.max_hvac_power, self.max_hvac_power, 24.0]),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_temp = 22.0
        self.heating_power = 0.0
        self.cooling_power = 0.0
        
        # Randomize outdoor temperature and time
        self.outdoor_temp = np.random.uniform(15.0, 35.0)
        self.time_of_day = np.random.uniform(0, 24)
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Take a step in the environment."""
        self.current_step += 1
        
        # Parse actions (normalized to [-1, 1])
        heating_action = (action[0] + 1) / 2  # Convert to [0, 1]
        cooling_action = (action[1] + 1) / 2  # Convert to [0, 1]
        
        # Calculate HVAC power
        self.heating_power = heating_action * self.max_hvac_power
        self.cooling_power = cooling_action * self.max_hvac_power
        
        # Update temperature based on HVAC and outdoor conditions
        temp_change = 0.0
        
        # Heating effect
        if self.current_temp < self.target_temp:
            temp_change += self.heating_power * 0.1
        
        # Cooling effect
        if self.current_temp > self.target_temp:
            temp_change -= self.cooling_power * 0.1
        
        # Outdoor temperature influence
        temp_diff = self.outdoor_temp - self.current_temp
        temp_change += temp_diff * 0.05
        
        # Update temperature
        self.current_temp += temp_change
        
        # Add some randomness
        self.current_temp += np.random.normal(0, 0.1)
        
        # Clamp temperature to reasonable range
        self.current_temp = np.clip(self.current_temp, 10.0, 35.0)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        """Get current observation."""
        return np.array([
            self.current_temp,
            self.outdoor_temp,
            self.heating_power,
            self.cooling_power,
            self.time_of_day
        ], dtype=np.float32)
    
    def _calculate_reward(self):
        """Calculate reward based on comfort and energy efficiency."""
        # Comfort reward (higher when temperature is in comfort range)
        temp_error = abs(self.current_temp - self.target_temp)
        comfort_reward = -temp_error * 2.0  # Penalty for temperature deviation
        
        # Energy efficiency reward (penalty for high energy consumption)
        total_energy = self.heating_power + self.cooling_power
        energy_penalty = -total_energy * 0.1
        
        # Bonus for being in comfort range
        if self.comfort_range[0] <= self.current_temp <= self.comfort_range[1]:
            comfort_reward += 1.0
        
        return comfort_reward + energy_penalty

# =============================================================================
# PPO TRAINING FUNCTIONS
# =============================================================================

def create_ppo_model(env):
    """Create a PPO model with good default parameters."""
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )
    return model

def train_ppo_agent(env, total_timesteps=50000):
    """Train a PPO agent on the environment."""
    print("Creating PPO model...")
    model = create_ppo_model(env)
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    return model

def evaluate_model(model, env, num_episodes=5):
    """Evaluate the trained model."""
    print(f"Evaluating model over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_energies = []
    episode_temps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_temp = []
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_energy += obs[2] + obs[3]  # heating + cooling power
            episode_temp.append(obs[0])  # current temperature
        
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_temps.append(np.mean(episode_temp))
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Energy = {episode_energy:.2f}, Avg Temp = {np.mean(episode_temp):.2f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'mean_temp': np.mean(episode_temps),
        'episode_rewards': episode_rewards,
        'episode_energies': episode_energies,
        'episode_temps': episode_temps
    }

def compare_with_random(env, num_episodes=5):
    """Compare trained model with random actions."""
    print(f"Testing random actions over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_energies = []
    episode_temps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_temp = []
        
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_energy += obs[2] + obs[3]  # heating + cooling power
            episode_temp.append(obs[0])  # current temperature
        
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
        episode_temps.append(np.mean(episode_temp))
        
        print(f"Random Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Energy = {episode_energy:.2f}, Avg Temp = {np.mean(episode_temp):.2f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_energy': np.mean(episode_energies),
        'mean_temp': np.mean(episode_temps),
        'episode_rewards': episode_rewards,
        'episode_energies': episode_energies,
        'episode_temps': episode_temps
    }

def plot_results(trained_results, random_results):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Reward comparison
    axes[0, 0].bar(['Trained', 'Random'], 
                   [trained_results['mean_reward'], random_results['mean_reward']])
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_ylabel('Reward')
    
    # Energy comparison
    axes[0, 1].bar(['Trained', 'Random'], 
                   [trained_results['mean_energy'], random_results['mean_energy']])
    axes[0, 1].set_title('Average Energy Consumption')
    axes[0, 1].set_ylabel('Energy (kW)')
    
    # Temperature comparison
    axes[1, 0].bar(['Trained', 'Random'], 
                   [trained_results['mean_temp'], random_results['mean_temp']])
    axes[1, 0].set_title('Average Temperature')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].axhline(y=22, color='r', linestyle='--', label='Target Temp')
    
    # Episode rewards
    axes[1, 1].plot(trained_results['episode_rewards'], label='Trained', marker='o')
    axes[1, 1].plot(random_results['episode_rewards'], label='Random', marker='s')
    axes[1, 1].set_title('Episode Rewards')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('ppo_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# MAIN DEMO FUNCTION
# =============================================================================

def run_ppo_demo():
    """Run the complete PPO demo."""
    print("="*60)
    print("PPO TRAINING DEMO (Without EnergyPlus)")
    print("="*60)
    print()
    print("This demo shows PPO training on a simple building environment.")
    print("The environment simulates:")
    print("- Temperature control")
    print("- Energy consumption")
    print("- Comfort violations")
    print("- Weather influence")
    print()
    
    # Create environment
    print("Creating environment...")
    env = SimpleBuildingEnv(max_steps=500)
    print(f"✓ Environment created:")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Max steps: {env.max_steps}")
    
    # Test environment
    print("\nTesting environment...")
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    
    # Take a few random steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action = {action}, Reward = {reward:.2f}, Temp = {obs[0]:.2f}")
    
    # Train PPO agent
    print("\n" + "="*40)
    print("TRAINING PPO AGENT")
    print("="*40)
    trained_model = train_ppo_agent(env, total_timesteps=20000)
    
    # Evaluate trained model
    print("\n" + "="*40)
    print("EVALUATING TRAINED MODEL")
    print("="*40)
    trained_results = evaluate_model(trained_model, env, num_episodes=5)
    
    # Compare with random
    print("\n" + "="*40)
    print("COMPARING WITH RANDOM ACTIONS")
    print("="*40)
    random_results = compare_with_random(env, num_episodes=5)
    
    # Show results
    print("\n" + "="*40)
    print("RESULTS COMPARISON")
    print("="*40)
    print(f"{'Metric':<20} {'Trained':<15} {'Random':<15} {'Improvement'}")
    print("-" * 60)
    
    reward_improvement = trained_results['mean_reward'] - random_results['mean_reward']
    energy_improvement = random_results['mean_energy'] - trained_results['mean_energy']
    temp_improvement = abs(trained_results['mean_temp'] - 22.0) - abs(random_results['mean_temp'] - 22.0)
    
    print(f"{'Average Reward':<20} {trained_results['mean_reward']:<15.2f} {random_results['mean_reward']:<15.2f} {reward_improvement:+.2f}")
    print(f"{'Energy Consumption':<20} {trained_results['mean_energy']:<15.2f} {random_results['mean_energy']:<15.2f} {energy_improvement:+.2f}")
    print(f"{'Temp Deviation':<20} {abs(trained_results['mean_temp']-22):<15.2f} {abs(random_results['mean_temp']-22):<15.2f} {temp_improvement:+.2f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(trained_results, random_results)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("You can now:")
    print("1. Run the full Sinergym tutorial with EnergyPlus")
    print("2. Experiment with different hyperparameters")
    print("3. Try different reward functions")
    print("4. Extend the environment with more features")
    
    return trained_model, trained_results, random_results

if __name__ == "__main__":
    run_ppo_demo()