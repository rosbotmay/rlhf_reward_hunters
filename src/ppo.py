import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gym
from stable_baselines3 import PPO


# Dataset and reward model
class PreferenceDataset(Dataset):
    def __init__(self, prefs):
        self.prefs = prefs

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx):
        return self.prefs[idx]


class RewardNet(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def compute_traj_return(reward_net, traj):
    """Compute cumulative reward for a trajectory."""
    device = next(reward_net.parameters()).device
    states = torch.tensor(
        [step[0] for step in traj], dtype=torch.float32, device=device
    )
    rewards = reward_net(states)
    return rewards.sum()


def train_reward_model(prefs, state_dim, device):
    reward_net = RewardNet(state_dim).to(device)
    opt = optim.Adam(reward_net.parameters(), lr=1e-3)

    dataset = PreferenceDataset(prefs)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)

    for epoch in range(10):
        print(f"[Reward Model] Epoch {epoch + 1} / 10")
        total_loss = 0.0
        for batch in loader:
            loss = 0.0
            for p in batch:
                tau1 = p["tau1"]
                tau2 = p["tau2"]
                pref = p["p_tau1_pref"] 

                ret1 = compute_traj_return(reward_net, tau1)
                ret2 = compute_traj_return(reward_net, tau2)

                logit = (ret1 - ret2).clamp(-10, 10)  # stability
                label = torch.tensor(pref, dtype=torch.float32, device=device)
                loss += nn.functional.binary_cross_entropy_with_logits(logit, label)

            loss /= len(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"[Reward Model] Epoch {epoch + 1}, Loss = {total_loss:.4f}")

    return reward_net


# wrap reward model to use in PPO
class RewardModelWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, device):
        super().__init__(env)
        self.reward_model = reward_model
        self.device = device

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            reward = float(self.reward_model(obs_tensor).item())
        return obs, reward, terminated, truncated, info


def main():
    print("Loading preferences...")
    with open("../data/prefs_expert_vs_subexpert.pkl", "rb") as f:
        prefs = pickle.load(f)
    prefs = prefs[:500]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 4

    print("using device:", device)
    reward_model = train_reward_model(prefs, state_dim, device)

    env = RewardModelWrapper(gym.make("CartPole-v1"), reward_model, device)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    model.save("ppo_rlhf_cartpole")


if __name__ == "__main__":
    main()
