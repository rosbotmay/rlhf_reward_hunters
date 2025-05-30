import os
import gym
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from reward import RewardModel

# 1) Define a one‐hot feature function for MountainCar
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from config import ENV_NAME,DATASIZES2,MODELS_DIR, RESULTS_DIR, SEEDS
from utils import evaluate_ppo_model

class RLHFCallBack(BaseCallback):
    """
    A callback that:
     - tracks per-episode shaped rewards and original episode lengths
     - prints rolling averages every `print_per_iter` episodes
     - saves model checkpoints every `save_freq` episodes (optionally when avg > threshold)
    """
    def __init__(
        self,
        save_freq: int = 10,
        avg_reward_window: int = 100,
        avg_reward_threshold: float = 0.0,
        print_per_iter: int = 10,
        save_checkpoint: bool = True,
        tag: str = "",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.avg_reward_window = avg_reward_window
        self.avg_reward_threshold = avg_reward_threshold
        self.print_per_iter = print_per_iter
        self.save_checkpoint = save_checkpoint
        self.tag = tag

        # buffers for shaped reward
        self._episode_rewards: list[float] = []
        self._current_ep_reward: float = 0.0

        # buffers for original reward (episode length)
        self._episode_lengths: list[int] = []
        self._current_ep_length: int = 0

        self._episode_count: int = 0

        # history of rolling averages
        self.avg_history: list[float] = []
        self.original_avg_history: list[float] = []

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]   # shaped rewards from env
        dones   = self.locals["dones"]     # done flags

        # accumulate for the first (and only) env
        self._current_ep_reward += float(rewards[0])
        self._current_ep_length += 1

        if dones[0]:
            # episode ended
            self._episode_count += 1

            # record this episode
            self._episode_rewards.append(self._current_ep_reward)
            self._episode_lengths.append(self._current_ep_length)

            # compute rolling averages
            recent_r = self._episode_rewards[-self.avg_reward_window :]
            recent_l = self._episode_lengths[-self.avg_reward_window :]
            shaped_avg = float(np.mean(recent_r))
            orig_avg   = - float(np.mean(recent_l))  # original reward is -1 per step

            # store histories
            self.avg_history.append(shaped_avg)
            self.original_avg_history.append(orig_avg)

            # print every print_per_iter episodes
            if (
                self.verbose
                and self._episode_count % self.print_per_iter == 0
                and len(self._episode_rewards) >= self.print_per_iter
            ):
                last_r = self._episode_rewards[-self.print_per_iter :]
                last_l = self._episode_lengths[-self.print_per_iter :]
                shaped_last_avg = float(np.mean(last_r))
                orig_last_avg   = - float(np.mean(last_l))
                print(
                    f"[RLHF] Episode {self._episode_count} | "
                    f"shaped avg ({self.print_per_iter}): {shaped_last_avg:.2f}, "
                    f"original avg ({self.print_per_iter}): {orig_last_avg:.2f}"
                )

            # checkpoint every save_freq episodes
            if self.save_checkpoint and (self._episode_count % self.save_freq == 0):
                if shaped_avg >= self.avg_reward_threshold:
                    fname = f"rlhf_{self.tag}_ep{self._episode_count}_avg{shaped_avg:.2f}.zip"
                    save_path = os.path.join(self.model.logger.dir, fname)
                    self.model.save(save_path)
                    if self.verbose:
                        print(f"[RLHF] Saved checkpoint: {save_path}")

            # reset counters for next episode
            self._current_ep_reward = 0.0
            self._current_ep_length = 0

        return True

    def _on_training_end(self) -> None:
        # final checkpoint
        if self.save_checkpoint:
            fname = f"rlhf_{self.tag}_final_ep{self._episode_count}.zip"
            save_path = os.path.join(self.model.logger.dir, fname)
            self.model.save(save_path)
            if self.verbose:
                print(f"[RLHF] Final model saved to {save_path}")



def feature_fn(obs, action):
    # obs: array-like of shape (obs_dim,)
    # action: a scalar (e.g. int or float)
    # 1) Make sure both are 1-D arrays
    obs_arr = np.asarray(obs).ravel()              # shape (obs_dim,)
    act_arr = np.atleast_1d(action).astype(float)  # shape (1,)
    # 2) Concatenate and flatten (flatten is redundant here but harmless)
    feat = np.concatenate([obs_arr, act_arr]).ravel()  # shape (obs_dim+1,)
    # 3) To PyTorch, add batch-dim
    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)


class RewardModelWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, feature_fn,gamma=0.99, lam=0.1):
        super().__init__(env)
        self.reward_model = reward_model.eval()
        self.feature_fn = feature_fn
        self.prev_score = 0.0
        self.lam = lam   # hyperparameter for shaping
        self.gamma = gamma  # discount factor for future rewards

    def reset(self, *args, **kwargs):
        # Pass through seed, options, etc.
        result = self.env.reset(*args, **kwargs)
        # Mirror tuple vs. single return
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
            
        # start cumulative model score at zero
        self.prev_score = 0.0
    
        return (obs, info) if isinstance(result, tuple) else obs

    def step(self, action):
        result = self.env.step(action)
        # gym ≥0.26 returns 5-tuple
        if len(result) == 5:
            obs, env_r, terminated, truncated, info = result
        else:  # old 4-tuple: (obs, reward, done, info)
            obs, env_r, done, info = result
            terminated, truncated = done, False
        # compute new preference score
        feats = self.feature_fn(obs, action)
        with torch.no_grad():
            score = float(self.reward_model(feats))
        # shaped reward = Δscore
        bonus = self.gamma * score - self.prev_score
        shaped_reward = env_r + self.lam * bonus
        self.prev_score = score
       # re-assemble the tuple in the same format
        
        return obs, shaped_reward, terminated, truncated, info


def run_episodes(env, model, n_episodes=30, render=False):
    all_returns = []
    all_lengths = []

    for ep in range(n_episodes):
        # reset returns either obs or (obs, info)
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        ep_return = 0.0
        length = 0

        while not done:
            # get deterministic action
            action, _ = model.predict(obs, deterministic=True)
            # step env: handle 5‐tuple vs 4‐tuple
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            ep_return += reward
            length += 1
            if render:
                env.render()

        all_returns.append(ep_return)
        all_lengths.append(length)

    return all_returns


# 4) Putting it all together
if __name__ == "__main__":
    
    for seed in SEEDS :

        for size in DATASIZES2:
            # load your pretrained reward model weights
            rm = RewardModel(input_dim=3)
            rm.load_state_dict(torch.load(f"{MODELS_DIR}reward_model_{size}.pth"))

            # base env
            base_env = gym.make(ENV_NAME)
            wrapped = RewardModelWrapper(base_env, rm, feature_fn)
            vec_env = DummyVecEnv([lambda: wrapped])
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

            callback = RLHFCallBack(
                save_freq=20,
                avg_reward_window=100,
                avg_reward_threshold=10.0,  
                print_per_iter=10,
                save_checkpoint=False,
                tag="mountaincar50",
                verbose=1,
            )

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
                ent_coef=0.01,
                seed=seed
            )

        model.learn(total_timesteps=100_000,callback=callback)
        model.save(f"{MODELS_DIR}ppo_rlhf/ppo_rlhf_mountaincar_{size}")

        shaped_avgs   = np.array(callback.avg_history)
        np.save(f"{RESULTS_DIR}ppo_rlhf/rlhf_mountaincar_seed_{seed}_{size}_shaped_avgs.npy", shaped_avgs)
        original_avgs = np.array(callback.original_avg_history)
        np.save(f"{RESULTS_DIR}ppo_rlhf/rlhf_mountaincar_seed_{seed}_{size}_original_avgs.npy", original_avgs)

        plt.plot(shaped_avgs,   label="Shaped reward")
        plt.plot(original_avgs, label="Original reward")
        plt.xlabel("Episode")
        plt.ylabel(f"Rolling mean over last {callback.avg_reward_window} eps")
        plt.legend()
        plt.title("Reward evolution during training")
        plt.savefig(f"{RESULTS_DIR}rlhf_mountaincar_seed_{seed}_{size}_rewards.png")

        # evaluation
        #model = PPO.load(f"{MODELS_DIR}ppo_rlhf/ppo_rlhf_mountaincar_{size}", env=vec_env)
        rewards = run_episodes(vec_env, model, n_episodes=100, render=False)
        np.save(f"{RESULTS_DIR}ppo_rlhf/ppo_rlhf_mountaincar_seed_{seed}_{size}_rewards.npy", np.array(rewards))

    
    

    
