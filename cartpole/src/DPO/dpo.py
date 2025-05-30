# src/dpo.py
import os
import pickle
import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import (
    BATCH_SIZE_DPO,
    CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    DPO_EPOCHS,
    DPO_LR,
    ENV_NAME,
)
from policy import Policy


class PrefDataset(Dataset):
    def __init__(self, prefs_list):
        self.prefs = prefs_list

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx):
        return self.prefs[idx]


def dpo_batch_loss(policy, batch):
    device = next(policy.parameters()).device
    losses = []
    for p in batch:
        logp1 = torch.tensor(0.0, device=device)
        logp2 = torch.tensor(0.0, device=device)

        # compute log probabilities of actions in trajectories
        # tau1: expert, tau2: subexpert
        for s, a, _ in p["tau1"]:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            # logp1 += Categorical(policy(s_t)).log_prob(torch.tensor(a, device=device)).sum()
            logp1 += (
                Categorical(policy(s_t))
                .log_prob(torch.tensor(a, device=device))
                .view(())
            )

        for s, a, _ in p["tau2"]:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            # logp2 += Categorical(policy(s_t)).log_prob(torch.tensor(a, device=device)).sum()
            logp2 += (
                Categorical(policy(s_t))
                .log_prob(torch.tensor(a, device=device))
                .view(())
            )

        # normalize by average trajectory length
        L1, L2 = len(p["tau1"]), len(p["tau2"])
        # norm = (L1 + L2) / 2 or 1
        norm = max((L1 + L2) / 2, 1)

        logit = (logp1 - logp2) / norm

        # compute loss
        # p_tau1_pref: probability of preferring tau1 over tau2
        # label: 1 if tau1 preferred, 0 otherwise
        target = torch.tensor(p["p_tau1_pref"], dtype=torch.float32, device=device)
        losses.append(F.binary_cross_entropy_with_logits(logit, target))
    return torch.stack(losses).mean()


def train_dpo(policy, prefs_file):
    # load prefs

    with open(os.path.join(DATA_DIR, prefs_file), "rb") as f:
        prefs = pickle.load(f)
    K = 700
    prefs = prefs[:K]

    # dataloader
    loader = DataLoader(
        PrefDataset(prefs),
        batch_size=BATCH_SIZE_DPO,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    # optimizer
    opt = torch.optim.Adam(policy.parameters(), lr=DPO_LR)

    policy.train()
    for ep in range(1, DPO_EPOCHS + 1):
        total_loss = 0.0
        for batch in loader:
            loss = dpo_batch_loss(policy, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"DPO Epoch {ep}/{DPO_EPOCHS}: loss = {total_loss / len(loader):.4f}")

    # save
    out = os.path.join(CHECKPOINT_DIR, "dpo_policy_expert_sub_700.pth")
    torch.save(policy.state_dict(), out)
    print(f"Saved DPO policy to {out}")
    return policy


def evaluate_dpo_policy(
    checkpoint_path: str, n_episodes: int = 50, seed = 0
) -> tuple[float, float]:
    """
    Load a saved DPO policy from `checkpoint_path` and
    return (mean_return, std_return) over `n_episodes` episodes.
    """
    # load policy
    env = gym.make(ENV_NAME)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]

    # for discrete envs, one-hot encode actions
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]

    policy = Policy(state_size=obs_dim, action_size=act_dim).to(DEVICE)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    policy.eval()

    # eval loop
    env = gym.make(ENV_NAME)
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = policy.act(obs)
            step = env.step(action)
            if len(step) == 5:
                obs, r, term, trunc, _ = step
                done = term or trunc
            else:
                obs, r, done, _ = step
            ep_ret += r
        returns.append(ep_ret)

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    return mean_r, std_r
