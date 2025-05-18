# src/ppo_rlhf.py
import gymnasium as gym  # Use Gymnasium instead of gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque
import argparse
import os
import pickle
from torch.utils.data import Dataset, DataLoader

# Placeholder for imported modules (assumed to exist)
from policy import Policy  # Define Policy class with state_size, action_size
from utils import action_to_tensor, make_pref_dataset, collect_bucket

# Config defaults (replace with config.py if available)
ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
TOTAL_STEPS = 100000
BATCH_SIZE = 64
EPOCHS_PER_BATCH = 10
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
LR_PI = 3e-4
LR_VF = 1e-3
EVAL_INTERVAL = 1000
N_EVAL_EPISODES = 10

# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

# Reward Model
class RewardModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Preference Dataset
class PrefDataset(Dataset):
    def __init__(self, prefs_list):
        self.prefs = prefs_list

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx):
        return self.prefs[idx]

# PPO-RLHF Agent
class PPO:
    def __init__(self, env, state_size, action_size, args):
        self.env = env
        self.device = DEVICE
        self.policy = Policy(state_size, action_size).to(self.device)
        self.value = ValueNetwork(state_size).to(self.device)
        self.reward_model = RewardModel(state_size, action_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR_PI)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LR_VF)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=args.reward_lr)
        self.args = args
        self.action_space = env.action_space

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        values = values + [next_value]  # Append next_value for the last step
        for r, v, d in zip(reversed(rewards), reversed(values[:-1]), reversed(dones)):
            delta = r + GAMMA * values[-1] * (1 - d) - v
            gae = delta + GAMMA * LAMBDA * (1 - d) * gae
            advantages.insert(0, gae)
            values.pop()  # Remove used next_value
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, states, actions, log_probs, advantages, returns):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        for _ in range(EPOCHS_PER_BATCH):
            probs = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()
            policy_loss = policy_loss - ENT_COEF * entropy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            value_pred = self.value(states).squeeze()
            value_loss = (value_pred - returns).pow(2).mean() * VF_COEF
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

#
def train_reward_model(agent, dataset, action_size):
    def collate_fn(batch):
        return batch

    loader = DataLoader(
        PrefDataset(dataset),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    for epoch in range(10):
        for batch in loader:
            losses = []
            for p in batch:
                try:
                    tau1_sum, tau2_sum = 0, 0
                    tau1_len, tau2_len = len(p['tau1']), len(p['tau2'])
                    if tau1_len == 0 or tau2_len == 0:
                        print(f"Warning: Empty trajectory detected (tau1_len={tau1_len}, tau2_len={tau2_len})")
                        continue
                    for s, a, _ in p['tau1']:
                        s_t = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
                        a_t = action_to_tensor(a, agent.action_space, action_size, DEVICE).unsqueeze(0)
                        tau1_sum += agent.reward_model(s_t, a_t).squeeze()
                    for s, a, _ in p['tau2']:
                        s_t = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
                        a_t = action_to_tensor(a, agent.action_space, action_size, DEVICE).unsqueeze(0)
                        tau2_sum += agent.reward_model(s_t, a_t).squeeze()
                    tau1_reward = tau1_sum / max(tau1_len, 1)
                    tau2_reward = tau2_sum / max(tau2_len, 1)
                    logit = tau1_reward - tau2_reward
                    target = torch.tensor(p['label'], dtype=torch.float32, device=DEVICE)
                    logit = logit.squeeze()
                    if logit.shape != target.shape or logit.dim() != 0:
                        print(f"Shape mismatch: logit={logit.shape}, target={target.shape}")
                        continue
                    loss = nn.functional.binary_cross_entropy_with_logits(logit, target)
                    losses.append(loss)
                except Exception as e:
                    print(f"Error in reward model training for pair: {e}")
                    continue
            if not losses:
                print("Warning: No valid losses computed for batch")
                continue
            loss = torch.stack(losses).mean()
            agent.reward_optimizer.zero_grad()
            loss.backward()
            agent.reward_optimizer.step()

def evaluate_policy(agent, env, n_episodes=N_EVAL_EPISODES):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0
        while not done:
            action, _ = agent.get_action(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += r
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))

def train_ppo_rlhf(env, args, seed, K, pi1_path, pi2_path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPO(env, state_size, action_size, args)
    
    # Load or generate preference dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    pref_file = os.path.join(DATA_DIR, f'prefs_expert_vs_subexpert_{K}_{seed}.pkl')
    try:
        if not os.path.exists(pref_file):
            print(f"Generating preference dataset for K={K}, seed={seed}")
            pi1 = Policy(state_size, action_size).to(DEVICE)
            pi2 = Policy(state_size, action_size).to(DEVICE)
            pi1.load_state_dict(torch.load(pi1_path, map_location=DEVICE))
            pi2.load_state_dict(torch.load(pi2_path, map_location=DEVICE))
            expert_trajs = collect_bucket(pi1, ENV_NAME, K)
            subexpert_trajs = collect_bucket(pi2, ENV_NAME, K)
            dataset = make_pref_dataset(expert_trajs, subexpert_trajs, seed)
            with open(pref_file, 'wb') as f:
                pickle.dump(dataset, f)
        else:
            with open(pref_file, 'rb') as f:
                dataset = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or generate preference dataset: {e}")
    
    # Train reward model
    train_reward_model(agent, dataset, action_size)
    
    # PPO-RLHF training
    step = 0
    episode = 0
    running_rewards = []
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    
    while step < TOTAL_STEPS:
        obs, _ = env.reset()
        ep_rewards = []
        done = False
        while not done:
            action, log_prob = agent.get_action(obs)
            value = agent.value(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)).item()
            a_one_hot = action_to_tensor(action, env.action_space, action_size, DEVICE).unsqueeze(0)
            r = agent.reward_model(
                torch.FloatTensor(obs).unsqueeze(0).to(DEVICE),
                a_one_hot
            ).item()
            obs, env_r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(r)
            values.append(value)
            dones.append(done)
            ep_rewards.append(r)
            step += 1
            
            if len(states) >= BATCH_SIZE or (done and states):
                next_value = agent.value(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)).item() if not done else 0
                advantages, returns = agent.compute_gae(rewards, values, next_value, dones)
                agent.update(states, actions, log_probs, advantages, returns)
                states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
            
            if step % EVAL_INTERVAL == 0:
                mean_r, std_r = evaluate_policy(agent, env)
                print(f"Seed: {seed}, K: {K}, Step: {step}, Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")
        
        if done:
            running_rewards.append(sum(ep_rewards))
            episode += 1
    
    # Save policy and reward model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(agent.policy.state_dict(), os.path.join(CHECKPOINT_DIR, f'ppo_rlhf_policy_K{K}_seed{seed}.pth'))
    torch.save(agent.reward_model.state_dict(), os.path.join(CHECKPOINT_DIR, f'ppo_rlhf_reward_K{K}_seed{seed}.pth'))
    return running_rewards, evaluate_policy(agent, env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_lr", type=float, default=1e-3, help="Reward model learning rate")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument("--pi1_path", type=str, default="C:/Users/ghrab/OneDrive/Bureau/Reinforcement Learning/rlhf_reward_hunters-CartPole/rlhf_reward_hunters-CartPole/policy_pi1.pth", help="Path to pi1 policy")
    parser.add_argument("--pi2_path", type=str, default="C:/Users/ghrab/OneDrive/Bureau/Reinforcement Learning/rlhf_reward_hunters-CartPole/rlhf_reward_hunters-CartPole/policy_pi2.pth", help="Path to pi2 policy")
    args = parser.parse_args()
    
    DATA_DIR = args.data_dir
    env = gym.make(ENV_NAME)
    seeds = [42, 43, 44]
    K_values = [100, 500, 1000]
    results = {}
    
    for seed in seeds:
        for K in K_values:
            try:
                rewards, (mean_r, std_r) = train_ppo_rlhf(env, args, seed, K, args.pi1_path, args.pi2_path)
                results[(seed, K)] = (rewards, mean_r, std_r)
                print(f"Final PPO-RLHF, Seed: {seed}, K: {K}, Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}")
            except Exception as e:
                print(f"Error for Seed: {seed}, K: {K}: {e}")
    
    # Save results
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'ppo_rlhf_results.pkl'), 'wb') as f:
        pickle.dump(results, f)