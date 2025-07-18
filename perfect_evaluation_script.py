import gymnasium as gym
import sinergym
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from sinergym.utils.rewards import LinearReward
from sinergym.utils.wrappers import (
    LoggerWrapper, CSVLogger, NormalizeObservation, NormalizeAction
)
from transformers import DistilBertTokenizer, DistilBertModel
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for professional plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============== BERT VARIANT SELECTION ==============
def get_bert_extractor(bert_mode='trainable'):
    """
    Returns the appropriate DistilBERT extractor based on training mode.
    
    Args:
        bert_mode (str): 'fixed', 'trainable', or 'partial'
    """
    
    if bert_mode == 'fixed':
        class DistilBERTExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, features_dim=128):
                super().__init__(observation_space, features_dim)
                self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
                self.bert.eval()  # Fixed - no training
                for param in self.bert.parameters():
                    param.requires_grad = False
                
                self.linear = nn.Sequential(
                    nn.Linear(self.bert.config.hidden_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, features_dim)
                )
            
            def forward(self, obs):
                input_ids = obs.squeeze(1).long()
                attention_mask = (input_ids != 0).long()
                with torch.no_grad():
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0]
                return self.linear(cls_embedding)
    
    elif bert_mode == 'partial':
        class DistilBERTExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, features_dim=128):
                super().__init__(observation_space, features_dim)
                self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
                self.bert.train()  # Partially trainable
                
                # Freeze some layers (e.g., first 4 layers)
                for i, layer in enumerate(self.bert.transformer.layer):
                    if i < 4:  # Freeze first 4 layers
                        for param in layer.parameters():
                            param.requires_grad = False
                
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
    
    else:  # trainable (default)
        class DistilBERTExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, features_dim=128):
                super().__init__(observation_space, features_dim)
                self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
                self.bert.train()  # Fully trainable
                
                self.linear = nn.Sequential(
                    nn.Linear(self.bert.config.hidden_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, features_dim)
                )
            
            def forward(self, obs):
                input_ids = obs.squeeze(1).long()
                attention_mask = (input_ids != 0).long()
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0]
                return self.linear(cls_embedding)
    
    return DistilBERTExtractor

# ============== CUSTOM POLICY ==============
def get_ppo_policy(bert_mode='trainable'):
    """Returns the appropriate PPO policy with the correct extractor."""
    DistilBERTExtractor = get_bert_extractor(bert_mode)
    
    class PPOBertPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs,
                             features_extractor_class=DistilBERTExtractor,
                             features_extractor_kwargs=dict(features_dim=128))
    
    return PPOBertPolicy

# ============== OBSERVATION WRAPPER ==============
class SentenceObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.observation_space = spaces.Box(low=0, high=self.tokenizer.vocab_size,
                                            shape=(1, 64), dtype=np.int32)
        self.raw_obs = None

    def observation(self, obs):
        self.raw_obs = obs.copy()  # Store raw observation
        sentence = (
            f"On day {int(obs[1])} of month {int(obs[0])} at {int(obs[2])}:00 hours, the outdoor temperature is {round(obs[3], 1)}°C "
            f"with humidity {int(obs[4])}%. Indoor temperature is {round(obs[11], 1)}°C and humidity is {int(obs[12])}%. "
            f"There are {int(obs[13])} people in the room with CO2 level at {round(obs[14], 1)} ppm. "
            f"HVAC is consuming {round(obs[15], 1)}W and total electricity usage is {round(obs[16], 1)}W."
        )
        encoded = self.tokenizer(sentence, padding='max_length', truncation=True,
                                 max_length=64, return_tensors="np")
        return encoded["input_ids"]

# ============== ENVIRONMENT SETUP ==============
def setup_environment(reward_kwargs, env_name='PPO-BERT-Evaluation'):
    """Setup environment with proper wrappers matching training."""
    env = gym.make('Eplus-5zone-hot-continuous-v1', config_params={
        'timesteps_per_hour': 1,
        'runperiod': (1, 1, 1991, 31, 12, 1991),
        'reward': reward_kwargs
    }, env_name=env_name)
    
    env = LoggerWrapper(env)
    env = CSVLogger(env)
    env = NormalizeObservation(env)  # ✅ Match training setup
    env = NormalizeAction(env)       # ✅ Match training setup
    env = SentenceObservationWrapper(env)
    env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))
    
    return env

# ============== MODEL LOADING ==============
def load_model(model_path, bert_mode='trainable'):
    """Load model with proper custom objects."""
    print(f"Loading model from: {model_path}")
    print(f"BERT mode: {bert_mode}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Get the appropriate policy
    PPOBertPolicy = get_ppo_policy(bert_mode)
    DistilBERTExtractor = get_bert_extractor(bert_mode)
    
    # Load model with custom objects
    model = PPO.load(
        model_path,
        custom_objects={
            "ActorCriticPolicy": PPOBertPolicy,
            "BaseFeaturesExtractor": DistilBERTExtractor
        }
    )
    
    print("✅ Model loaded successfully!")
    return model

# ============== EVALUATION FUNCTION ==============
def run_evaluation(model, env, num_episodes=12):
    """Run evaluation episodes and collect detailed metrics."""
    all_results = []
    
    print(f"Running {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        obs, _ = env.reset()
        terminated = truncated = False
        episode_results = []
        step = 0
        
        while not (terminated or truncated):
            # Capture raw observation BEFORE stepping
            current_raw_obs = env.raw_obs.copy()
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_results.append({
                'Episode': episode + 1,
                'Step': step,
                'Month': current_raw_obs[0],
                'Day': current_raw_obs[1],
                'Hour': current_raw_obs[2],
                'Indoor Temperature': current_raw_obs[11],
                'Outdoor Temperature': current_raw_obs[3],
                'Energy Usage': current_raw_obs[15],
                'Reward': reward
            })
            step += 1
        
        all_results.extend(episode_results)
        print(f"Episode {episode + 1} completed - Steps: {step}")
    
    return all_results

# ============== STATISTICS & PLOTTING ==============
def calculate_monthly_stats(df):
    """Calculate monthly statistics with proper confidence intervals."""
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

def plot_evaluation_graphs(monthly_stats, lambda_temperature, lambda_energy, energy_weight, 
                          bert_mode, save_dir):
    """Create professional evaluation plots."""
    
    def plot_graph(y_label, mean_col, sd_col, ci_col, title, y_lim, y_ticks, filename):
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
        
        plt.title(f'{title} - {bert_mode.upper()} BERT', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=14, fontweight='bold')
        plt.ylabel(y_label, fontsize=14, fontweight='bold')
        plt.ylim(y_lim)
        plt.yticks(y_ticks, fontsize=12)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                 fontsize=12, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, framealpha=0.9)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot: {plot_path}")
    
    # Create plots
    plot_graph('Indoor Temperature (°C)', 'Indoor_Temp_Mean', 'Indoor_Temp_SD', 'Indoor_Temp_CI',
               f'Indoor Temperature (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
               (18, 27), np.arange(18, 28.5, 0.5), f'indoor_temperature_{bert_mode}.png')

    plot_graph('Outdoor Temperature (°C)', 'Outdoor_Temp_Mean', 'Outdoor_Temp_SD', 'Outdoor_Temp_CI',
               f'Outdoor Temperature (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
               (5, 40), range(5, 41, 2), f'outdoor_temperature_{bert_mode}.png')

    plot_graph('Energy Usage (W)', 'Energy_Usage_Mean', 'Energy_Usage_SD', 'Energy_Usage_CI',
               f'Energy Usage (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
               (0, 5000), range(0, 5001, 250), f'energy_usage_{bert_mode}.png')

    plot_graph('Reward', 'Reward_Mean', 'Reward_SD', 'Reward_CI',
               f'Reward (λt={lambda_temperature}, λe={lambda_energy}, w={energy_weight})', 
               (-10, 2), np.arange(-10, 2.5, 1), f'reward_{bert_mode}.png')

# ============== MAIN EVALUATION FUNCTION ==============
def main():
    """Main evaluation function with command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate PPO DistilBERT models')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the trained model')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the model file (e.g., ppo_distilbert_20250709-025523.zip)')
    parser.add_argument('--bert_mode', type=str, default='trainable',
                       choices=['fixed', 'trainable', 'partial'],
                       help='BERT training mode used during training')
    parser.add_argument('--num_episodes', type=int, default=12,
                       help='Number of evaluation episodes')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    print("🚀 Starting PPO DistilBERT Model Evaluation")
    print("=" * 60)
    print(f"Model Directory: {args.model_dir}")
    print(f"Model Name: {args.model_name}")
    print(f"BERT Mode: {args.bert_mode}")
    print(f"Number of Episodes: {args.num_episodes}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load reward configuration
        reward_files = list(Path(args.model_dir).glob('reward_*.json'))
        if not reward_files:
            raise FileNotFoundError(f"No reward configuration file found in {args.model_dir}")
        
        latest_reward_file = max(reward_files, key=lambda x: x.stat().st_mtime)
        with open(latest_reward_file, 'r') as f:
            reward_kwargs = json.load(f)
        
        print(f"📋 Loaded reward configuration from: {latest_reward_file}")
        
        # Extract reward parameters
        lambda_temperature = reward_kwargs.get('lambda_temperature', 28)
        lambda_energy = reward_kwargs.get('lambda_energy', 0.01)
        energy_weight = reward_kwargs.get('energy_weight', 0.4)
        
        # Setup environment
        print("\n🔧 Setting up evaluation environment...")
        env = setup_environment(reward_kwargs, f'PPO-BERT-{args.bert_mode}-Evaluation')
        
        # Load model
        model_path = os.path.join(args.model_dir, args.model_name)
        model = load_model(model_path, args.bert_mode)
        
        # Run evaluation
        print(f"\n🎯 Running evaluation with {args.num_episodes} episodes...")
        results = run_evaluation(model, env, args.num_episodes)
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        csv_path = os.path.join(args.output_dir, f'ppo_bert_evaluation_{args.bert_mode}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
        
        # Calculate monthly statistics
        print("\n📊 Calculating monthly statistics...")
        monthly_stats = calculate_monthly_stats(df)
        monthly_csv_path = os.path.join(args.output_dir, f'monthly_stats_{args.bert_mode}.csv')
        monthly_stats.to_csv(monthly_csv_path, index=False)
        
        # Create plots
        print("\n📈 Creating evaluation plots...")
        plot_evaluation_graphs(monthly_stats, lambda_temperature, lambda_energy, 
                             energy_weight, args.bert_mode, args.output_dir)
        
        # Print summary statistics
        print("\n📋 Evaluation Summary:")
        print("=" * 40)
        print(f"Total Episodes: {args.num_episodes}")
        print(f"Total Data Points: {len(df)}")
        print(f"Average Indoor Temperature: {df['Indoor Temperature'].mean():.2f}°C ± {df['Indoor Temperature'].std():.2f}°C")
        print(f"Average Energy Usage: {df['Energy Usage'].mean():.2f}W ± {df['Energy Usage'].std():.2f}W")
        print(f"Average Reward: {df['Reward'].mean():.2f} ± {df['Reward'].std():.2f}")
        
        # Comfort zone analysis
        indoor_temps = df['Indoor Temperature']
        winter_comfort = ((indoor_temps >= 20.0) & (indoor_temps <= 23.5)).mean() * 100
        summer_comfort = ((indoor_temps >= 23.0) & (indoor_temps <= 26.0)).mean() * 100
        
        print(f"\n🏠 Comfort Zone Analysis:")
        print(f"Winter Comfort (20-23.5°C): {winter_comfort:.1f}% of time")
        print(f"Summer Comfort (23-26°C): {summer_comfort:.1f}% of time")
        
        # Close environment
        env.close()
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 All results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    # Example usage:
    # python perfect_evaluation_script.py --model_dir /path/to/model --model_name ppo_distilbert_20250709-025523.zip --bert_mode trainable
    
    # For interactive use without command line arguments, uncomment and modify the lines below:
    """
    # Manual configuration (uncomment and modify as needed)
    MODEL_DIR = '/Users/z5543337/Desktop/work/PPO-DistilBERT-trainable-20250709-025523-res1'
    MODEL_NAME = 'ppo_distilbert_20250709-025523.zip'
    BERT_MODE = 'trainable'  # 'fixed', 'trainable', or 'partial'
    NUM_EPISODES = 12
    
    # Set up arguments manually
    import sys
    sys.argv = [
        'perfect_evaluation_script.py',
        '--model_dir', MODEL_DIR,
        '--model_name', MODEL_NAME,
        '--bert_mode', BERT_MODE,
        '--num_episodes', str(NUM_EPISODES)
    ]
    """
    
    main()