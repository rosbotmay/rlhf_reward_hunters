from policy import Policy
from config import ENV_NAME, DEVICE, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR
import torch
from rlhf_reward_hunters.src.DPO.dpo import evaluate_dpo_policy
import os
import numpy as np

if __name__ == "__main__":
    Ks = [500, 600, 700, 800, 900, 1000]
    episodes = 100
    seeds = [45, 65, 76]

    for K in Ks:
        dpo_ckpt  = os.path.join(CHECKPOINT_DIR, f"dpo_policy_expert_sub_{K}.pth")
        pi2_ckpt  = os.path.join(DATA_DIR,     "policy_pi2.pth")

        # results per seed
        dpo_returns, pi2_returns = [], []

        for seed in seeds:
            # evaluate DPO policy under this seed
            mean_dpo, std_dpo = evaluate_dpo_policy(dpo_ckpt, n_episodes=episodes, seed=seed)
            dpo_returns.append(mean_dpo)

            # evaluate the original pi2 policy under the same seed
            mean_pi2, std_pi2 = evaluate_dpo_policy(pi2_ckpt, n_episodes=episodes, seed=seed)
            pi2_returns.append(mean_pi2)

        # compute across-seed averages and stds
        avg_dpo   = np.mean(dpo_returns)
        std_dpo   = np.std(dpo_returns)
        avg_pi2   = np.mean(pi2_returns)
        std_pi2   = np.std(pi2_returns)

        # write out a single summary per K
        out_path = os.path.join(RESULTS_DIR, f"results_dpo_{K}.txt")
        with open(out_path, "w") as f:
            f.write(f"K = {K}\n")
            f.write(f"pre-DPO pi2 (over seeds) : mean return = {avg_pi2:.2f} ± {std_pi2:.2f}\n")
            f.write(f"DPO (over seeds)        : mean return = {avg_dpo:.2f} ± {std_dpo:.2f}\n\n")
            f.write(f"Individual seed returns:\n")
            for s, (r2, rd) in enumerate(zip(pi2_returns, dpo_returns)):
                f.write(f"  seed={seeds[s]}: pi2 = {r2:.2f}, DPO = {rd:.2f}\n")

        print(f"[K={K}] pi2: {avg_pi2:.2f}±{std_pi2:.2f}, DPO: {avg_dpo:.2f}±{std_dpo:.2f}")