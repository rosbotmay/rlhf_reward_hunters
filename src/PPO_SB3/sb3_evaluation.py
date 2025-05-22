import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import CHECKPOINT_DIR, ENV_NAME

def evaluate(model, env, n_episodes=50):
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        rewards.append(ep_reward)

    rewards = np.array(rewards)
    mean = rewards.mean()
    std_err = rewards.std(ddof=1) / np.sqrt(len(rewards))
    ci95 = 1.96 * std_err

    return mean, ci95

# evaluation environments
eval_env_subexpert = gym.make(ENV_NAME)
eval_env_rlhf = gym.make(ENV_NAME)

# load models
model_subexpert = PPO.load(f"{CHECKPOINT_DIR}\\ppo_subexpert", env=eval_env_subexpert)
model_rlhf = PPO.load(f"{CHECKPOINT_DIR}\\ppo_rlhf_finetuned_v1", env=eval_env_rlhf)

# evaluate both
mean_sub, ci_sub = evaluate(model_subexpert, eval_env_subexpert, n_episodes=30)
mean_rlhf, ci_rlhf = evaluate(model_rlhf, eval_env_rlhf, n_episodes=30)

# output results
print(f"📊 Subexpert: {mean_sub:.2f} ± {ci_sub:.2f} (95% CI)")
print(f"🚀 PPO-RLHF: {mean_rlhf:.2f} ± {ci_rlhf:.2f} (95% CI)")
print(f"🎯 Absolute improvement: {mean_rlhf - mean_sub:.2f}")
print(f"📈 Relative improvement: {(mean_rlhf - mean_sub) / (500 - mean_sub) * 100:.2f}%")
