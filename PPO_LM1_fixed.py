import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import (
    LoggerWrapper, CSVLogger, NormalizeObservation, NormalizeAction
)
from sinergym.utils.rewards import LinearReward
from sinergym.utils.callbacks import LoggerEvalCallback
from stable_baselines3 import PPO
from stable_baselines3n.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
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
import time

# ======== Tokenizer =========
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ======== Sentence Obs Wrapper =========
class SentenceObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tokenizer = tokenizer
        self.observation_space = spaces.Box(low=0, high=tokenizer.vocab_size,
                                            shape=(1,64), dtype=np.int32)

    def observation(self, obs):
        sentence = (
            f"On day {int(obs[1])} of month {int(obs[0])} at {int(obs[2])} hours, the outdoor temperature is {round(obs[3], 1)}°C "
            f"with humidity {int(obs[4])}%. Indoor temperature is {round(obs[11], 1)}°C and humidity is {int(obs[12])}%. "
            f"There are {int(obs[13])} people in the room with CO2 level at {round(obs[14], 1)} ppm. "
            f"HVAC is consuming {round(obs[15], 2)} total electricity usage is {round(obs[16], 2)}"
        )
        encoded = self.tokenizer(sentence, padding='max_length', truncation=True,
                                 max_length=64, return_tensors="np")
        return encoded["input_ids"]

# ======== DistilBERT Extractor with Profiling =========
class DistilBERTExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert.to("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.train()  # Fully trainable
        self.linear = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim)
        )

    def forward(self, obs):
        device = next(self.linear.parameters()).device
        obs = obs.squeeze(1).long().to(device)
        attention_mask = (obs != 0).long().to(device)

        start = time.time()
        outputs = self.bert(input_ids=obs, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # CLS token
        end = time.time()

        # Save timing (append to file)
        with open("profiling_log.csv", "a") as f:
            f.write(f"distilbert,{end-start}\n")
        return self.linear(cls_embedding)

# ======== Custom Policy =========
class PPOBertPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=DistilBERTExtractor,
                         features_extractor_kwargs=dict(features_dim=128))

# ======== Reward Setup =========
lambda_temperature = 28
lambda_energy = 0.01
energy_weight = 0.4
reward_kwargs = {
    "temperature_variables": ["air_temperature"],
    "energy_variables": ["HVAC_electricity_demand_rate"],
    "range_comfort_winter": [20, 23.5],
    "range_comfort_summer": [23, 26],
    "summer_start": [6, 1],
    "summer_final": [9, 30],
    "energy_weight": energy_weight,
    "lambda_energy": lambda_energy,
    "lambda_temperature": lambda_temperature
}

# ======== Environment Setup =========
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
experiment_name = f'PPO-DistilBERT-trainable-{timestamp}'
env_id = 'Eplus-5zone-hot-continuous-v1'

# FIXED:Create environment without passing config_params and env_name to gym.make()
# These parameters should be configured differently
train_env = gym.make(env_id)
train_env = LoggerWrapper(train_env)
train_env = CSVLogger(train_env)
train_env = NormalizeObservation(train_env)
train_env = NormalizeAction(train_env)
train_env = SentenceObservationWrapper(train_env)

# Set the reward function
train_env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))

class TimingWrapper(gym.Wrapper):
    def step(self, action):
        start = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        end = time.time()
        with open("profiling_log.csv", "a") as f:
            f.write(f"sinergym_step,{end-start}\n")
        return obs, reward, terminated, truncated, info

train_env = TimingWrapper(train_env)  # profile sinergym step

# Eval env
eval_env = gym.make(env_id)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = SentenceObservationWrapper(eval_env)
eval_env.set_wrapper_attr('reward_fn', LinearReward(**reward_kwargs))

# ======== PPO Model =========
model = PPO(
    policy=PPOBertPolicy,
    env=train_env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,
    verbose=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# ======== Callback =========
eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=train_env,
    n_eval_episodes=3,
    eval_freq_episodes=2,
    deterministic=True
)
callback = CallbackList([eval_callback])

# ======== Train =========
episodes = 3
total_timesteps = episodes * train_env.get_wrapper_attr('timestep_per_episode')
with open("profiling_log.csv", "w") as f:
    f.write("component,time_sec\n")  # CSV header
model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=500)

# ======== Save Model =========
workspace = train_env.get_wrapper_attr('workspace_path')
os.makedirs(workspace, exist_ok=True)
model_path = os.path.join(workspace, f'ppo_distilbert_{timestamp}')
model.save(model_path)
with open(os.path.join(workspace, 'reward.json'), 'w') as f:
    json.dump(reward_kwargs, f, indent=4)

# ======== Plot Reward Curve =========
progress_path = os.path.join(workspace, 'progress.csv')
progress_df = pd.read_csv(progress_path)
plt.figure(figsize=(8, 6))
plt.plot(progress_df['episode_num'], progress_df['mean_reward'], marker='o')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('PPO + DistilBERT Reward Convergence')
plt.grid(True)
plt.savefig(os.path.join(workspace, f'reward_plot_{timestamp}.png'))
plt.show()