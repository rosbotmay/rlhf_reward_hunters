import gym
from scipy.stats import pearsonr
import torch
import numpy as np
from config import DEVICE, CHECKPOINT_DIR, ENV_NAME
from reward_model import RewardModel


def evaluate_reward_model():
    eval_env = gym.make(ENV_NAME)
    n_eval = 20
    true_returns, pred_returns = [], []
    eval_env.reset(seed=0)
    obs_dim = eval_env.observation_space.shape[0]

    # for discrete envs, one-hot encode actions
    if isinstance(eval_env.action_space, gym.spaces.Discrete):
        act_dim = eval_env.action_space.n
    else:
        act_dim = eval_env.action_space.shape[0]

    rm = RewardModel(obs_dim=obs_dim, act_dim=act_dim, hidden=64).to(DEVICE)
    state = torch.load(f"{CHECKPOINT_DIR}\\reward_model.pth", map_location=DEVICE)
    rm.load_state_dict(state)

    with torch.no_grad():
        for _ in range(n_eval):
            reset_out = eval_env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            R_true, R_hat = 0.0, 0.0
            while not done:
                # sample a random action for pure env rollout
                action = eval_env.action_space.sample()
                step_out = eval_env.step(action)
                if len(step_out) == 5:
                    next_obs, r, term, trunc, _ = step_out
                    done = term or trunc
                else:
                    next_obs, r, done, _ = step_out

                R_true += r

                # now predict reward_model
                # first convert obs to tensor efficiently
                obs_arr = np.array(obs, dtype=np.float32)         # shape (4,)
                s_t    = torch.as_tensor(obs_arr, device=DEVICE)  # 1D float32 tensor
                s_t    = s_t.unsqueeze(0)                         # shape (1,4)

                # one-hot action
                a_t = torch.zeros((1, act_dim), device=DEVICE)
                a_t[0, action] = 1.0

                with torch.no_grad():
                    r_hat = rm(s_t, a_t).item()
                R_hat += r_hat

                obs = next_obs

                true_returns.append(R_true)
                pred_returns.append(R_hat)

    corr = np.corrcoef(true_returns, pred_returns)[0,1]
    print(f"True mean={np.mean(true_returns):.1f},  Pred mean={np.mean(pred_returns):.1f},  Pearson r={corr:.2f}")
    
if __name__ == "__main__":
    evaluate_reward_model()
