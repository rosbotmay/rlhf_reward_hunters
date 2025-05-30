import os
import pickle
import gymnasium as gym
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from config import CHECKPOINT_PAIRS, DATASIZES2, ENV_NAME, SEEDS
from utils import evaluate_ppo_model
import numpy as np

# === 1. Policy wrapper to extract log-probs ===
class PolicyWrapper(torch.nn.Module):
    def __init__(self, sb3_model):
        super().__init__()
        self.policy = sb3_model.policy

    def forward(self, obs, actions):
        # obs: [B, obs_dim], actions: [B]
        dist = self.policy.get_distribution(obs)
        logp = dist.log_prob(actions)
        return logp

# === 2. Dataset for preference pairs ===
class PreferenceDataset(Dataset):
    def __init__(self, pairs):
        # Load list of (traj1, traj2, prob) tuples
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        traj1, traj2, prob = self.pairs[idx]
        # Convert trajectories to tensors
        obs1 = torch.tensor([s for (s,a,_,_,_) in traj1], dtype=torch.float32)
        act1 = torch.tensor([a for (s,a,_,_,_) in traj1], dtype=torch.long)
        obs2 = torch.tensor([s for (s,a,_,_,_) in traj2], dtype=torch.float32)
        act2 = torch.tensor([a for (s,a,_,_,_) in traj2], dtype=torch.long)
        # convert probability to logit
        logit_p = torch.logit(torch.tensor(prob, dtype=torch.float32), eps=1e-6)
        return obs1, act1, obs2, act2, logit_p

# === 3. Collate function ===
def collate_batch(batch):
    obs1, act1, obs2, act2, logits = zip(*batch)
    return obs1, act1, obs2, act2, torch.stack(logits)

# === 4. DPO training loop without reward model ===
def train_dpo(
    ref_sb3,
    tgt_sb3: str,
    preferences: list,
    batch_size: int = 4,
    epochs: int = 3,
    beta: float = 1.0,
    lr: float = 1e-5,
    device: str = 'cpu',
    tag=""
):
    
    ref_policy = PolicyWrapper(ref_sb3).to(device).eval()
    tgt_policy = PolicyWrapper(tgt_sb3).to(device).train()

    dataset = PreferenceDataset(preferences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    optimizer = torch.optim.Adam(tgt_policy.parameters(), lr=lr)
    optimal_count = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for obs1_list, act1_list, obs2_list, act2_list, logit_p in loader:
            losses = []
            for obs1, act1, obs2, act2, lp in zip(obs1_list, act1_list, obs2_list, act2_list, logit_p):
                obs1, act1 = obs1.to(device), act1.to(device)
                obs2, act2 = obs2.to(device), act2.to(device)
                lp = lp.to(device)

                # log-prob under target and ref
                sum_lp1 = tgt_policy(obs1, act1).sum()
                sum_lp2 = tgt_policy(obs2, act2).sum()
                sum_lr1 = ref_policy(obs1, act1).sum().detach()
                sum_lr2 = ref_policy(obs2, act2).sum().detach()
                delta_logp = (sum_lp1 - sum_lr1) - (sum_lp2 - sum_lr2)

                # DPO loss: -logσ(delta_logp + β * logit(p))
                loss = -F.logsigmoid(delta_logp + beta * lp)
                losses.append(loss)

            batch_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

            

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

        if optimal_count >= 10:
            break

        if epoch_loss/len(loader) == 0.0:
            optimal_count += 1

    # Save fine-tuned policy
    tgt_sb3.policy = tgt_policy.policy
    out_path = f"models/dpo_finetuned_policy_{tag}.zip"
    tgt_sb3.save(out_path)
    print(f"DPO-finetuned policy saved to {out_path}")
    return tgt_sb3

# Function to set the seed for Gymnasium environments
def set_seed_for_env(env, seed):
    env.action_space.seed(seed)  # Set the seed for action space
    env.reset(seed=seed)         # Set the seed for the environment reset

# === 5. Example usage ===
if __name__ == '__main__':
    
    ref_ckpt = CHECKPOINT_PAIRS[0][0]
    tgt_ckpt = CHECKPOINT_PAIRS[0][1]

    
    for seed in SEEDS:
        
        env = gym.make(ENV_NAME)
        set_seed_for_env(env, seed)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        # Load reference and target SB3 models
        ref_sb3 = PPO.load(ref_ckpt,env=vec_env)
        ref_rewards = evaluate_ppo_model(ref_sb3,vec_env,episodes=100)
        tgt_sb3 = PPO.load(tgt_ckpt,env=vec_env)
        r_pre_dpo = evaluate_ppo_model(tgt_sb3,vec_env,episodes=100)
        np.save(f"results/dpo/pre_dpo_rewards_seed_{seed}.npy", np.array(r_pre_dpo)) ## seed dependent

        pref_pickle = "data/preference_pairs_2000_original.pkl"
        with open(pref_pickle, 'rb') as f:
            pref_data = pickle.load(f)

        #500, 1e-5 
        for size in  DATASIZES2:
            preferences = pref_data[:size]
            ft_sb3 = train_dpo(ref_sb3, tgt_sb3, preferences,
                    batch_size=2, epochs=100, beta=1.0, lr=1e-5, device='cpu',tag=f"seed_{seed}_{size}")
            r_post_dpo = evaluate_ppo_model(ft_sb3, vec_env, episodes=20)
            np.save(f"results/dpo/post_dpo_rewards_seed_{seed}_{size}.npy", np.array(r_post_dpo))

            #Save rewards histograms
            plt.figure(figsize=(10, 6))
            # Plot histograms for pi1, pi2, and pi2_ft
            plt.hist(ref_rewards, bins=20, alpha=0.5, label="pi1 (Optimal Policy)")
            plt.hist(r_pre_dpo, bins=20, alpha=0.5, label="pi2 (Pre-DPO Policy)")
            plt.hist(r_post_dpo, bins=20, alpha=0.5, label="pi2 (Post-DPO Policy)")

            plt.xlabel("Total Reward")
            plt.ylabel("Frequency")
            plt.title("Total Rewards for pi1, pi2, and pi2_ft over 100 episodes")
            plt.legend(loc='upper left')
            # Save the histogram plot
            plt.savefig(f"results/dpo/reward_histogram_seed_{seed}_size_{size}.png")
            plt.close() 
