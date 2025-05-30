import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from config import ENV_NAME, CHECKPOINTS_DIR, RESULTS_DIR


# Set the seed for reproducibility
seed_n = 42
np.random.seed(seed_n)
rewards_list = list()
# Define the number of episodes and how often to print
episodes = 1500
print_per_iter = 10

# Define the callback for saving the model based on rolling average reward of last 100 episodes
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0, save_freq=10, avg_reward_window=100, avg_reward_threshold=0,save_checkpoint=True):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.curr_episode = 0
        self.verbose = True
        self.save_freq = save_freq
        self.avg_reward_window = avg_reward_window
        self.avg_reward_threshold = avg_reward_threshold
        self.save_checkpoint = save_checkpoint

    def _on_step(self) -> bool:
        global rewards_list, print_per_iter

        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_episode_reward += rewards[0]

        # Print the total reward at the end of an episode
        if dones[0]:
            if self.verbose and self.curr_episode % print_per_iter == 0 and self.curr_episode != 0:
                avg_reward = np.mean(rewards_list[-print_per_iter:])
                print(f"Episode {self.curr_episode}, Reward: {avg_reward:.2f}")
                            
            self.episode_rewards.append(self.current_episode_reward)
            rewards_list.append(self.current_episode_reward)
            self.current_episode_reward = 0
            self.curr_episode += 1

            # Save the model based on the rolling average of the last `avg_reward_window` rewards
            if len(rewards_list) >= self.avg_reward_window:
                avg_reward_last_window = np.mean(rewards_list[-self.avg_reward_window:])
                if self.curr_episode % self.save_freq == 0 and self.save_checkpoint :
                    # Save checkpoint when avg reward surpasses threshold
                    print(f"Saving model checkpoint at episode {self.curr_episode} with avg reward {avg_reward_last_window:.2f}")
                    self.model.save(f"{CHECKPOINTS_DIR}/ppo/ppo_checkpoint_{self.curr_episode}_avg_reward_{avg_reward_last_window:.2f}")

        return True

# Plot the results and evaluate the model
def compute_rolling_average(rewards, window_size=print_per_iter):
    return [
        np.mean(rewards[max(0, i - window_size + 1):i + 1])
        for i in range(len(rewards))
    ]

def plot_single_reward(rewards, title, window_size=print_per_iter):
    avg_rewards = compute_rolling_average(rewards, window_size=window_size)
        
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards, label="Rolling Average Reward")  # Adding label for legend
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()  # This will now show the legend
    plt.grid()
    plt.savefig(f"{RESULTS_DIR}ppo/mountain_car_evolution.png")
    plt.close()
    

if __name__ == "__main__":

    # Create the environment and set the seed
    env = gym.make(ENV_NAME)

    def set_rand_env(env, seed=seed_n):
        env.action_space.seed(seed)
        state, _ = env.reset(seed=seed)

    set_rand_env(env)

    # Wrap the environment for Stable Baselines3 with the Monitor wrapper
    env = Monitor(env)  # Wrap with Monitor to track episode length and rewards
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=32,
        gae_lambda=0.9,
        gamma=0.99,
        n_epochs=20,
        clip_range=0.2,
        seed=seed_n
    )

    # Train the model with the callback for saving checkpoints based on average reward of last 100 episodes
    model.learn(total_timesteps=episodes*200, callback=EpisodeRewardCallback(save_freq=50, avg_reward_window=100, avg_reward_threshold=-100))
    model.save(f"{CHECKPOINTS_DIR}/ppo/ppo_final_model")

    # Plot the rewards
    plot_single_reward(rewards_list, "PPO")
    np.save(f"{RESULTS_DIR}ppo/mountain_car_rewards.npy", np.array(rewards_list))

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=100, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
