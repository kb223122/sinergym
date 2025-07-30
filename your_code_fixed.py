import gymnasium as gym
import sinergym
from stable_baselines3 import PPO

def verify_parameters_at_step(env, step_num):
    """Verify parameters at each step."""
    print(f"\nStep {step_num}:")
    print(f"  Energy weight: {env.unwrapped.reward.W_energy}")
    print(f"  Temperature penalty: {env.unwrapped.reward.lambda_temp}")
    print(f"  Runperiod: {env.get_wrapper_attr('runperiod')}")

# ✅ CORRECT: Use 'extra_config' instead of 'config_params'
extra_config = {
    # Time period: Summer months only
    'runperiod': (1, 6, 1991, 31, 8, 1991),  # June 1 to August 31, 1991
    
    # Simulation resolution
    'timesteps_per_hour': 4
}

# ✅ CORRECT: Custom reward function parameters
reward_kwargs = {
    'temperature_variables': ['air_temperature'],
    'energy_variables': ['HVAC_electricity_demand_rate'],
    'range_comfort_winter': (20.0, 23.5),
    'range_comfort_summer': (23.0, 26.0),
    'summer_start': (6, 1),
    'summer_final': (8, 31),
    'energy_weight': 0.7,        # Focus on energy efficiency
    'lambda_energy': 0.0001,
    'lambda_temperature': 0.8     # Moderate comfort penalty
}

# ✅ CORRECT: Create environment with correct parameter name
env = gym.make('Eplus-5zone-hot-continuous-v1', 
               extra_config=extra_config,  # ← Added back!
               reward_kwargs=reward_kwargs)

# Print configuration to verify parameters were applied
print(f"Episode length: {env.get_wrapper_attr('timestep_per_episode')} timesteps")
print(f"Runperiod: {env.get_wrapper_attr('runperiod')}")

# Verify reward function parameters
reward_func = env.unwrapped.reward  # ← Fixed: use 'reward' not 'reward_fn'
print(f"Reward function type: {type(reward_func).__name__}")
if hasattr(reward_func, 'W_energy'):
    print(f"Energy weight: {reward_func.W_energy}")
if hasattr(reward_func, 'lambda_temp'):
    print(f"Temperature penalty: {reward_func.lambda_temp}")

# Test environment step by step to verify parameters
print(f"\n{'='*80}")
print(f"STEP-BY-STEP PARAMETER VERIFICATION")
print(f"{'='*80}")

obs, info = env.reset()
episode_reward = 0

for step in range(10):  # Test first 10 steps
    # Take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_reward += reward
    
    # Verify parameters at each step
    verify_parameters_at_step(env, step + 1)
    
    if terminated or truncated:
        print(f"Episode ended after {step + 1} steps")
        break

print(f"\nTotal episode reward: {episode_reward:.4f}")

# Train PPO agent episode-wise
# Calculate timesteps for 3 episodes
timesteps_per_episode = env.get_wrapper_attr('timestep_per_episode')
total_timesteps = 3 * timesteps_per_episode  # 3 episodes

print(f"\nTraining for 3 episodes ({total_timesteps} timesteps)")

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=total_timesteps)

print("Training completed!")
env.close()