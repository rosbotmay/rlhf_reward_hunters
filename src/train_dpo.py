# run_dpo.py
import torch
from config import ENV_NAME, DEVICE, CHECKPOINT_DIR, DATA_DIR
from policy import Policy
from dpo import train_dpo, evaluate_dpo_policy
import os

if __name__ == "__main__":
    # initialize & load pretrained policy
    policy = Policy(state_size=4, action_size=2).to(DEVICE)
    policy.load_state_dict(torch.load(f"{DATA_DIR}/policy_pi2.pth"))
    
    # fine-tune via DPO
    train_dpo(policy, prefs_file ="prefs_expert_vs_subexpert.pkl")

    ckpt = os.path.join(CHECKPOINT_DIR, "dpo_policy_expert_sub.pth")
    mean_r, std_r = evaluate_dpo_policy(ckpt, n_episodes=50)
    mean_r_pi2, std_r_pi2 = evaluate_dpo_policy(f"{DATA_DIR}/policy_pi2.pth", n_episodes=50)
    print(f"pre-DPO Policy: mean return={mean_r_pi2:.2f}, std={std_r_pi2:.2f} over 50 episodes")
    print(f"DPO Policy: mean return={mean_r:.2f}, std={std_r:.2f} over 50 episodes")
