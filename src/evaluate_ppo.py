from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO.load("ppo_rlhf_cartpole")

total_rewards = []
for _ in range(100):
    obs, _ = env.reset()
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
print(
    f"PPO-RLHF Policy: mean return = {mean_r:.2f}, std = {std_r:.2f} over 100 episodes"
)
