import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import CHECKPOINT_DIR

# Custom callback to stop training once average reward crosses 250
class StopTrainingCallback(BaseCallback):
    def __init__(self, threshold=250.0, eval_env=None, verbose=1):
        super().__init__(verbose)
        self.threshold = threshold
        self.eval_env = eval_env or gym.make("CartPole-v1", render_mode=None)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            episode_rewards = []

            for _ in range(5):  # average over 5 episodes
                obs, _ = self.eval_env.reset()
                done = False
                total_reward = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                episode_rewards.append(total_reward)

            avg_reward = np.mean(episode_rewards)
            if self.verbose:
                print(f"Step {self.n_calls}: Average reward = {avg_reward:.2f}")

            if avg_reward >= self.threshold:
                print("Target average reward reached. Stopping training.")
                return False

        return True

# Create CartPole environment (Gymnasium version)
env = gym.make("CartPole-v1", render_mode=None)

# Train the PPO subexpert
model = PPO("MlpPolicy", env, verbose=0)

callback = StopTrainingCallback(threshold=250.0)
model.learn(total_timesteps=100_000, callback=callback)

# ✅ Save the model
model.save(f"{CHECKPOINT_DIR}\\ppo_subexpert")
print("✅ Subexpert saved to 'ppo_subexpert.zip'")
