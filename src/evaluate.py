from policy import Policy
from config import ENV_NAME, DEVICE, CHECKPOINT_DIR, DATA_DIR
import torch
from dpo import evaluate_dpo_policy
import os

if __name__ == "__main__":
    # Load the DPO policy
    ckpt = os.path.join(CHECKPOINT_DIR, "dpo_policy_expert_sub_K500_seed1.pth")   
    mean_r, std_r = evaluate_dpo_policy(ckpt, n_episodes=50)
    mean_r_pi2, std_r_pi2 = evaluate_dpo_policy(f"{DATA_DIR}/policy_pi2.pth", n_episodes=50)   # comppare the performance of policy 2 and the after-DPO policy 
    print(f"pre-DPO Policy: mean return={mean_r_pi2:.2f}, std={std_r_pi2:.2f} over 50 episodes")
    print(f"DPO Policy: mean return={mean_r:.2f}, std={std_r:.2f} over 50 episodes")