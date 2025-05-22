import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import CHECKPOINT_DIR, DATA_DIR, ENV_NAME, DEVICE
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# ----------------------------
# Reward Model Definition
# ----------------------------
class RewardModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = F.relu(self.fc1(x))
        return self.out(F.relu(self.fc2(h))).squeeze(-1)

# ----------------------------
# Load Environment and Reward Model
# ----------------------------
env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
env.close()

rm = RewardModel(obs_dim, act_dim).to(DEVICE)
rm.load_state_dict(torch.load(f"{CHECKPOINT_DIR}\\reward_model.pth", map_location=DEVICE))
rm.eval()

# ----------------------------
# Reward Wrapper for Gymnasium
# ----------------------------
class RewardModelWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, device="cpu"):
        super().__init__(env)
        self.reward_model = reward_model.to(device)
        self.device = device
        self.act_dim = env.action_space.n  # for one-hot encoding

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        # One-hot encode the discrete action
        action_tensor = torch.zeros((1, self.act_dim), dtype=torch.float32).to(self.device)
        action_tensor[0, action] = 1.0

        with torch.no_grad():
            reward = self.reward_model(obs_tensor, action_tensor).cpu().item()

        return obs, reward, terminated, truncated, info

# ----------------------------
# PPO Training
# ----------------------------
def make_env():
    base_env = gym.make(ENV_NAME)
    return RewardModelWrapper(base_env, rm, device=DEVICE)

vec_env = make_vec_env(make_env, n_envs=4)

# Load subexpert and fine-tune
model = PPO.load(f"{CHECKPOINT_DIR}\\ppo_subexpert", env=vec_env)
model.learn(total_timesteps=500_000)
model.save("ppo_rlhf_finetuned")
print("✅ PPO RLHF model saved to 'ppo_rlhf_finetuned_v1.zip'")

# ----------------------------
# Evaluation
# ----------------------------
obs = vec_env.reset()
episode_reward = 0

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    episode_reward += reward
    if done.any():
        break

print(f"Final episode reward (RLHF): {episode_reward.mean():.2f}")
