from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import torch
import random

seeds = [45, 65, 76]

for seed in seeds:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    model = PPO.load("../models/ppo_rlhf_cartpole_600")

    total_rewards = []
    for _ in range(100):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)

    mean_r = sum(total_rewards) / len(total_rewards)
    std_r = (sum([(r - mean_r) ** 2 for r in total_rewards]) / len(total_rewards)) ** 0.5

    print(f"seed : {seed} PPO-RLHF Policy: mean return = {mean_r:.2f}, std = {std_r:.2f} over 100 episodes")
