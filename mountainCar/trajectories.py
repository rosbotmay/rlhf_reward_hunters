import math
import os
from config import ENV_NAME, CHECKPOINT_PAIRS
import pickle
import random
import numpy as np
import random
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils import evaluate_ppo_model


def generate_trajectory_ppo(model, env, max_timesteps=200):
    state = env.reset()  # Reset the environment and get the initial state
    done = False
    trajectory = []
    total_reward = 0
    while not done and len(trajectory) < max_timesteps:
        # Use the PPO model to predict the next action
        action, _ = model.predict(state, deterministic=True)  # Use deterministic action selection (exploitation)
        
        # Step the environment
        next_state, reward, done,info = env.step(action)
        
        # Store the trajectory (state, action, reward, next_state, done)
        trajectory.append((state.flatten(), action, reward, next_state, done))
        total_reward += reward
        
        # Update the state
        state = next_state
        
    return trajectory, total_reward
    

def main(env=gym.make(ENV_NAME),tag=""):

    for K in [500,2000]:
        
        preference_pairs = []
        K_by_pairs = K // len(CHECKPOINT_PAIRS)
        remaining = K % len(CHECKPOINT_PAIRS)
        for i,(pi1,pi2) in enumerate(CHECKPOINT_PAIRS):
            if i == len(CHECKPOINT_PAIRS) - 1:
                K_by_pairs += remaining

            agent_pi1 = PPO.load(pi1)
            agent_pi2 = PPO.load(pi2)

            rewards_pi1 = []
            rewards_pi2 = []

            for _ in range(K_by_pairs):
                # Generate a trajectory from policy pi1 and pi2
                trajectory_pi1, reward_pi1 = generate_trajectory_ppo(agent_pi1, env)
                trajectory_pi2, reward_pi2 = generate_trajectory_ppo(agent_pi2, env)

                rewards_pi1.append(reward_pi1)
                rewards_pi2.append(reward_pi2)
                print(reward_pi1)
                print(reward_pi2)

                # Compute the probability that pi1's trajectory is preferred
                prob_pi1 = math.exp(reward_pi1) / (math.exp(reward_pi1) + math.exp(reward_pi2))
                preference_pairs.append((trajectory_pi1, trajectory_pi2, prob_pi1))


        random.shuffle(preference_pairs)
        # Save the preference pairs
        with open(f"data/preference_pairs_{K}_{tag}.pkl", "wb") as f:
            pickle.dump(preference_pairs, f)
        print(f"Preference pairs saved to data/preference_pairs_{K}_{tag}.pkl")


if __name__ == "__main__":
    
    env = gym.make(ENV_NAME)
    env = Monitor(env)  # Wrap with Monitor to track episode length and rewards
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    for agent1, agent2 in CHECKPOINT_PAIRS:
        evaluate_ppo_model(PPO.load(agent1), env, episodes=100, render=False)

    main(env=env,tag="original")
    