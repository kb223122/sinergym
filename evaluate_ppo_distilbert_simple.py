import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import (
    LoggerWrapper, CSVLogger, NormalizeAction
)
from sinergym.utils.rewards import LinearReward
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from transformers import DistilBertTokenizer, DistilBertModel
from gymnasium import spaces
import torch
import torch.nn as nn
import numpy as np
import os
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for better plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Sentence Observation Wrapper (same as training)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class SentenceObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tokenizer = tokenizer
        self.observation_space = spaces.Box(low=0, high=tokenizer.vocab_size,
                                            shape=(1, 64), dtype=np.int32)

    def observation(self, obs):
        sentence = (
            f"On day {int(obs[1])} of month {int(obs[0])} at {int(obs[2])}:00 hours, the outdoor temperature is {round(obs[3], 1)}°C "
            f"with humidity {int(obs[4])}%. Indoor temperature is {round(obs[11], 1)}°C and humidity is {int(obs[12])}%. "
            f"There are {int(obs[13])} people in the room with CO2 level at {round(obs[14], 1)} ppm. "
            f"HVAC is consuming {round(obs[15], 1)}W and total electricity usage is {round(obs[16], 1)}W."
        )
        encoded = self.tokenizer(sentence, padding='max_length', truncation=True,
                                 max_length=64, return_tensors="np")
        return encoded["input_ids"]

# DistilBERT Feature Extractor (same as training)
class DistilBERTExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert.train()  # Fully trainable

        self.linear = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, obs):
        input_ids = obs.squeeze(1).long()
        attention_mask = (input_ids != 0).long()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return self.linear(cls_embedding)

# Custom Policy (same as training)
class PPOBertPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=DistilBERTExtractor,
                         features_extractor_kwargs=dict(features_dim=128))

def setup_environment(env_id, reward_kwargs, experiment_name):
    """Setup evaluation environment with same configuration as training"""
    extra_conf = {
        'timesteps_per_hour': 1,
        'runperiod': (1, 1, 1991, 31, 12, 1991),
        'reward': reward_kwargs
    }
    
    env = gym.make(env_id, config_params=extra_conf, env_name=experiment_name)
    env = LoggerWrapper(env)
    env = CSVLogger(env)
    env = NormalizeAction(env)
    env = SentenceObservationWrapper(env)
    env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))
    
    return env

def run_evaluation(model, env, num_episodes=12):
    """Run evaluation episodes and collect detailed metrics"""
    results = []
    all_episode_data = []
    
    print(f"Starting evaluation with {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"Running episode {episode + 1}/{num_episodes}")
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_data = []
        
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Extract relevant metrics from observation
            month = int(obs[0][0]) if hasattr(obs, 'shape') else int(obs[0])
            outdoor_temp = obs[3] if hasattr(obs, 'shape') else obs[3]
            indoor_temp = obs[11] if hasattr(obs, 'shape') else obs[11]
            energy_usage = obs[15] if hasattr(obs, 'shape') else obs[15]  # HVAC electricity
            
            episode_data.append({
                'Episode': episode + 1,
                'Step': step,
                'Month': month,
                'Outdoor Temperature': outdoor_temp,
                'Indoor Temperature': indoor_temp,
                'Energy Usage': energy_usage,
                'Reward': reward
            })
            
            episode_reward += reward
            step += 1
            
            if truncated:
                break
        
        # Add episode summary to results
        episode_df = pd.DataFrame(episode_data)
        results.append({
            'Episode': episode + 1,
            'Total Steps': len(episode_data),
            'Total Reward': episode_reward,
            'Mean Indoor Temp': episode_df['Indoor Temperature'].mean(),
            'Std Indoor Temp': episode_df['Indoor Temperature'].std(),
            'Mean Outdoor Temp': episode_df['Outdoor Temperature'].mean(),
            'Std Outdoor Temp': episode_df['Outdoor Temperature'].std(),
            'Mean Energy Usage': episode_df['Energy Usage'].mean(),
            'Std Energy Usage': episode_df['Energy Usage'].std(),
            'Mean Reward': episode_df['Reward'].mean(),
            'Std Reward': episode_df['Reward'].std()
        })
        
        all_episode_data.extend(episode_data)
        print(f"Episode {episode + 1} completed - Total Reward: {episode_reward:.2f}")
    
    return results, all_episode_data

def create_monthly_stats(df):
    """Create monthly statistics with confidence intervals"""
    monthly_stats = df.groupby('Month').agg({
        'Indoor Temperature': ['mean', 'std', 'count'],
        'Outdoor Temperature': ['mean', 'std', 'count'],
        'Energy Usage': ['mean', 'std', 'count'],
        'Reward': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    monthly_stats.columns = ['Month', 'Indoor_Temp_Mean', 'Indoor_Temp_SD', 'Indoor_Temp_Count',
                            'Outdoor_Temp_Mean', 'Outdoor_Temp_SD', 'Outdoor_Temp_Count',
                            'Energy_Usage_Mean', 'Energy_Usage_SD', 'Energy_Usage_Count',
                            'Reward_Mean', 'Reward_SD', 'Reward_Count']
    
    # Calculate 95% confidence intervals (z = 1.96)
    z = 1.96
    for metric in ['Indoor_Temp', 'Outdoor_Temp', 'Energy_Usage', 'Reward']:
        mean_col = f'{metric}_Mean'
        sd_col = f'{metric}_SD'
        count_col = f'{metric}_Count'
        ci_col = f'{metric}_CI'
        
        # Calculate standard error and confidence interval
        monthly_stats[f'{metric}_SE'] = monthly_stats[sd_col] / np.sqrt(monthly_stats[count_col])
        monthly_stats[ci_col] = z * monthly_stats[f'{metric}_SE']
    
    return monthly_stats

def plot_evaluation_graphs(monthly_stats, lambda_temperature, lambda_energy, energy_weight, save_path):
    """Create the 4 evaluation graphs with confidence intervals"""
    
    def plot_graph(y_label, mean_col, sd_col, ci_col, title, y_lim, y_ticks, x_ticks, filename):
        plt.figure(figsize=(12, 8))
        
        # Add comfort zone lines for indoor temperature
        if 'Indoor' in y_label:
            plt.axhline(y=20.0, color='darkgreen', linestyle='--', linewidth=2, 
                       label='Winter Min (20°C)', alpha=0.7)
            plt.axhline(y=23.5, color='darkgreen', linestyle='--', linewidth=2, 
                       label='Winter Max (23.5°C)', alpha=0.7)
            plt.axhline(y=23.0, color='darkorange', linestyle='-.', linewidth=2, 
                       label='Summer Min (23°C)', alpha=0.7)
            plt.axhline(y=26.0, color='darkorange', linestyle='-.', linewidth=2, 
                       label='Summer Max (26°C)', alpha=0.7)
        
        # Plot mean line
        plt.plot(monthly_stats['Month'], monthly_stats[mean_col], '-o', 
                linewidth=3, markersize=8, label='Mean', color='#2E86AB')
        
        # Plot standard deviation band
        plt.fill_between(monthly_stats['Month'],
                        monthly_stats[mean_col] - monthly_stats[sd_col],
                        monthly_stats[mean_col] + monthly_stats[sd_col], 
                        alpha=0.3, label='±SD', color='#A23B72')
        
        # Plot confidence interval band
        plt.fill_between(monthly_stats['Month'],
                        monthly_stats[mean_col] - monthly_stats[ci_col],
                        monthly_stats[mean_col] + monthly_stats[ci_col], 
                        alpha=0.2, label='95% CI', color='#F18F01')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=14, fontweight='bold')
        plt.ylabel(y_label, fontsize=14, fontweight='bold')
        plt.ylim(y_lim)
        plt.yticks(y_ticks, fontsize=12)
        plt.xticks(x_ticks, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, framealpha=0.9)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot: {plot_path}")
    
    # Create plots
    plot_graph('Indoor Temperature (°C)', 'Indoor_Temp_Mean', 'Indoor_Temp_SD', 'Indoor_Temp_CI',
               f'Indoor Temperature - PPO DistilBERT (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
               (18, 27), np.arange(18, 28.5, 0.5), range(1, 13), 'indoor_temperature_evaluation.png')

    plot_graph('Outdoor Temperature (°C)', 'Outdoor_Temp_Mean', 'Outdoor_Temp_SD', 'Outdoor_Temp_CI',
               f'Outdoor Temperature - PPO DistilBERT (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
               (5, 40), range(5, 41, 2), range(1, 13), 'outdoor_temperature_evaluation.png')

    plot_graph('Energy Usage (W)', 'Energy_Usage_Mean', 'Energy_Usage_SD', 'Energy_Usage_CI',
               f'Energy Usage - PPO DistilBERT (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
               (0, 5000), range(0, 5001, 250), range(1, 13), 'energy_usage_evaluation.png')

    plot_graph('Reward', 'Reward_Mean', 'Reward_SD', 'Reward_CI',
               f'Reward - PPO DistilBERT (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
               (-10, 2), np.arange(-10, 2.5, 1), range(1, 13), 'reward_evaluation.png')

def main():
    """Main evaluation function"""
    print("🚀 Starting PPO DistilBERT Model Evaluation")
    print("=" * 60)
    
    # Configuration - MODIFY THESE PATHS AS NEEDED
    MODEL_PATH = input("Enter the path to your trained model: ").strip()
    WORKSPACE_PATH = input("Enter the path to your training workspace: ").strip()
    
    # Reward configuration (same as your training)
    lambda_temperature = 28
    lambda_energy = 0.01
    energy_weight = 0.4
    reward_kwargs = {
        "temperature_variables": ["air_temperature"],
        "energy_variables": ["HVAC_electricity_demand_rate"],
        "range_comfort_winter": [20.0, 23.5],
        "range_comfort_summer": [23.0, 26.0],
        "summer_start": [6, 1],
        "summer_final": [9, 30],
        "energy_weight": energy_weight,
        "lambda_energy": lambda_energy,
        "lambda_temperature": lambda_temperature
    }
    
    env_id = 'Eplus-5zone-hot-continuous-v1'
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    evaluation_name = f'PPO-DistilBERT-Evaluation-{timestamp}'
    
    # Load model
    try:
        print(f"🤖 Loading model from: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=None, custom_objects={
            'ActorCriticPolicy': PPOBertPolicy,
            'BaseFeaturesExtractor': DistilBERTExtractor
        })
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please check the model path and ensure the model exists.")
        return
    
    # Setup evaluation environment
    print("\n🔧 Setting up evaluation environment...")
    eval_env = setup_environment(env_id, reward_kwargs, evaluation_name)
    
    # Run evaluation
    print("\n🎯 Running evaluation...")
    results, episode_data = run_evaluation(model, eval_env, num_episodes=12)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    episode_df = pd.DataFrame(episode_data)
    
    # Create evaluation directory
    eval_dir = os.path.join(WORKSPACE_PATH, f'evaluation_{timestamp}')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save results
    results_df.to_csv(os.path.join(eval_dir, "ppo_evaluation_summary.csv"), index=False)
    episode_df.to_csv(os.path.join(eval_dir, "ppo_evaluation_stepwise_metrics.csv"), index=False)
    
    print(f"\n💾 Results saved to: {eval_dir}")
    
    # Create monthly statistics
    print("\n📊 Creating monthly statistics...")
    monthly_stats = create_monthly_stats(episode_df)
    monthly_stats.to_csv(os.path.join(eval_dir, "monthly_statistics.csv"), index=False)
    
    # Create evaluation plots
    print("\n📈 Creating evaluation plots...")
    plot_evaluation_graphs(monthly_stats, lambda_temperature, lambda_energy, energy_weight, eval_dir)
    
    # Print summary statistics
    print("\n📋 Evaluation Summary:")
    print("=" * 40)
    print(f"Total Episodes: {len(results)}")
    print(f"Average Total Reward: {results_df['Total Reward'].mean():.2f} ± {results_df['Total Reward'].std():.2f}")
    print(f"Average Indoor Temperature: {results_df['Mean Indoor Temp'].mean():.2f}°C ± {results_df['Mean Indoor Temp'].std():.2f}°C")
    print(f"Average Energy Usage: {results_df['Mean Energy Usage'].mean():.2f}W ± {results_df['Mean Energy Usage'].std():.2f}W")
    print(f"Average Step Reward: {results_df['Mean Reward'].mean():.2f} ± {results_df['Mean Reward'].std():.2f}")
    
    # Comfort zone analysis
    indoor_temps = episode_df['Indoor Temperature']
    winter_comfort = ((indoor_temps >= 20.0) & (indoor_temps <= 23.5)).mean() * 100
    summer_comfort = ((indoor_temps >= 23.0) & (indoor_temps <= 26.0)).mean() * 100
    
    print(f"\n🏠 Comfort Zone Analysis:")
    print(f"Winter Comfort (20-23.5°C): {winter_comfort:.1f}% of time")
    print(f"Summer Comfort (23-26°C): {summer_comfort:.1f}% of time")
    
    # Close environment
    eval_env.close()
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 All results saved to: {eval_dir}")

if __name__ == "__main__":
    main()