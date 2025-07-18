import gymnasium as gym
import sinergym
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from sinergym.utils.rewards import LinearReward
from sinergym.utils.wrappers import LoggerWrapper, CSVLogger, NormalizeAction
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from transformers import DistilBertTokenizer, DistilBertModel
from gymnasium import spaces
import torch.nn as nn

# ✅ Sentence Observation Wrapper (EXACTLY same as training)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class SentenceObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tokenizer = tokenizer
        self.observation_space = spaces.Box(low=0, high=tokenizer.vocab_size,
                                            shape=(1, 64), dtype=np.int32)
        self.raw_obs = None  # Store raw observation for evaluation

    def observation(self, obs):
        self.raw_obs = obs  # Store raw observation before tokenization
        sentence = (
            f"On day {int(obs[1])} of month {int(obs[0])} at {int(obs[2])}:00 hours, the outdoor temperature is {round(obs[3], 1)}°C "
            f"with humidity {int(obs[4])}%. Indoor temperature is {round(obs[11], 1)}°C and humidity is {int(obs[12])}%. "
            f"There are {int(obs[13])} people in the room with CO2 level at {round(obs[14], 1)} ppm. "
            f"HVAC is consuming {round(obs[15], 1)}W and total electricity usage is {round(obs[16], 1)}W."
        )
        encoded = self.tokenizer(sentence, padding='max_length', truncation=True,
                                 max_length=64, return_tensors="np")
        return encoded["input_ids"]

# ✅ DistilBERT Feature Extractor (EXACTLY same as training)
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

# ✅ Custom Policy (EXACTLY same as training)
class PPOBertPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=DistilBERTExtractor,
                         features_extractor_kwargs=dict(features_dim=128))

# ⚙️ Setup Reward & Env (EXACTLY same as training)
model_dir = '/Users/z5543337/Downloads/'
model_name = 'PPO-DistilBERT-Partial-20250709-031717-res1/ppo_distilbert_20250709-031717.zip'

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

# ✅ Environment Setup (EXACTLY same as training)
env = gym.make('Eplus-5zone-hot-continuous-v1', config_params={
    'timesteps_per_hour': 1,
    'runperiod': (1, 1, 1991, 31, 12, 1991),
    'reward': reward_kwargs
}, env_name='PPO-LLM-Test')

env = LoggerWrapper(env)
env = CSVLogger(env)
env = NormalizeAction(env)  # ✅ Match training setup
env = SentenceObservationWrapper(env)
env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))

# ✅ Load Model with Custom Objects (CRITICAL FIX)
print("Loading model...")
model = PPO.load(os.path.join(model_dir, model_name), 
                 custom_objects={
                     'ActorCriticPolicy': PPOBertPolicy,
                     'BaseFeaturesExtractor': DistilBERTExtractor
                 })
print("Model loaded successfully!")

# ▶️ Run Multiple Episodes for Better Evaluation
num_episodes = 12  # One episode per month for full year coverage
all_results = []

for episode in range(num_episodes):
    print(f"Running episode {episode + 1}/{num_episodes}")
    
    obs, _ = env.reset()
    terminated = truncated = False
    episode_results = []
    step = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # ✅ Access raw observation correctly
        raw_obs = env.raw_obs
        
        episode_results.append({
            'Episode': episode + 1,
            'Step': step,
            'Month': raw_obs[0],
            'Day': raw_obs[1],
            'Hour': raw_obs[2],
            'Indoor Temperature': raw_obs[11],
            'Outdoor Temperature': raw_obs[3],
            'Energy Usage': raw_obs[15],
            'Reward': reward
        })
        step += 1
    
    all_results.extend(episode_results)
    print(f"Episode {episode + 1} completed - Steps: {step}")

env.close()

# 💾 Save CSV
df = pd.DataFrame(all_results)
df.to_csv("ppo_test_stepwise_metrics.csv", index=False)
print(f"Saved {len(df)} data points to CSV")

# 📊 Monthly Stats + Plotting
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

# ✅ Calculate confidence intervals correctly
z = 1.96
for metric in ['Indoor_Temp', 'Outdoor_Temp', 'Energy_Usage', 'Reward']:
    mean_col = f'{metric}_Mean'
    sd_col = f'{metric}_SD'
    count_col = f'{metric}_Count'
    ci_col = f'{metric}_CI'
    
    # Calculate standard error and confidence interval
    monthly_stats[f'{metric}_SE'] = monthly_stats[sd_col] / np.sqrt(monthly_stats[count_col])
    monthly_stats[ci_col] = z * monthly_stats[f'{metric}_SE']

# ✅ Enhanced plotting function
def plot_graph(y_label, mean_col, sd_col, ci_col, title, y_lim, y_ticks, x_ticks):
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
    plt.show()

# ✅ Create all 4 plots
plot_graph('Indoor Temperature (°C)', 'Indoor_Temp_Mean', 'Indoor_Temp_SD', 'Indoor_Temp_CI',
           f'Indoor Temperature - PPO DistilBERT (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
           (18, 27), np.arange(18, 28.5, 0.5), range(1, 13))

plot_graph('Outdoor Temperature (°C)', 'Outdoor_Temp_Mean', 'Outdoor_Temp_SD', 'Outdoor_Temp_CI',
           f'Outdoor Temperature - PPO DistilBERT (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
           (5, 40), range(5, 41, 2), range(1, 13))

plot_graph('Energy Usage (W)', 'Energy_Usage_Mean', 'Energy_Usage_SD', 'Energy_Usage_CI',
           f'Energy Usage - PPO DistilBERT (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
           (0, 5000), range(0, 5001, 250), range(1, 13))

plot_graph('Reward', 'Reward_Mean', 'Reward_SD', 'Reward_CI',
           f'Reward - PPO DistilBERT (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
           (-10, 2), np.arange(-10, 2.5, 1), range(1, 13))

# ✅ Print summary statistics
print("\n📋 Evaluation Summary:")
print("=" * 40)
print(f"Total Episodes: {num_episodes}")
print(f"Total Data Points: {len(df)}")
print(f"Average Indoor Temperature: {df['Indoor Temperature'].mean():.2f}°C ± {df['Indoor Temperature'].std():.2f}°C")
print(f"Average Energy Usage: {df['Energy Usage'].mean():.2f}W ± {df['Energy Usage'].std():.2f}W")
print(f"Average Reward: {df['Reward'].mean():.2f} ± {df['Reward'].std():.2f}")

# ✅ Comfort zone analysis
indoor_temps = df['Indoor Temperature']
winter_comfort = ((indoor_temps >= 20.0) & (indoor_temps <= 23.5)).mean() * 100
summer_comfort = ((indoor_temps >= 23.0) & (indoor_temps <= 26.0)).mean() * 100

print(f"\n🏠 Comfort Zone Analysis:")
print(f"Winter Comfort (20-23.5°C): {winter_comfort:.1f}% of time")
print(f"Summer Comfort (23-26°C): {summer_comfort:.1f}% of time")

print("\n✅ Evaluation completed successfully!")