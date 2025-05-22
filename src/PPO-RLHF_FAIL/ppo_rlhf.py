import gym
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch import nn
from config import ENV_NAME, DEVICE, LR_PI, GAMMA, LAM, PPO_EPOCHS, BATCH_SIZE, MINI_BATCH, CLIP_EPS, LR_V, DATA_DIR, CHECKPOINT_DIR
from reward_model import RewardModel
from policy import Policy
from train_value import ValueNet

torch.autograd.set_detect_anomaly(True)

# ——— GAE helper ———
def compute_gae(rewards, values, masks):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        next_val = values[t+1] if t+1 < len(values) else 0.0
        delta = rewards[t] + GAMMA * next_val * masks[t] - values[t]
        adv[t] = lastgaelam = delta + GAMMA * LAM * masks[t] * lastgaelam
    rets = adv + values[:T]
    return adv, rets

# ——— Collect batch under learned reward ———
def collect_batch():
    states, actions, old_logps = [], [], []
    rewards, masks, values      = [], [], []

    s = env.reset()
    if isinstance(s, tuple): s = s[0]
    done = False

    while len(states) < BATCH_SIZE:
        s_t  = torch.from_numpy(np.array(s, dtype=np.float32)).to(DEVICE).unsqueeze(0)
        probs= policy(s_t)
        dist = Categorical(probs)
        a    = dist.sample()
        lp   = dist.log_prob(a)
        v    = value(s_t).item()

        # step env
        step = env.step(a.item())
        if len(step)==5:
            s_next, _, term, trunc, _ = step; done = term or trunc
        else:
            s_next, _, done, _ = step

        # model reward
        a_ohe = torch.zeros((1,act_dim), device=DEVICE)
        a_ohe[0,a] = 1.0
        with torch.no_grad():
            r_pred = rm(s_t, a_ohe).item()

        # record
        states.append(s_t)
        actions.append(a)
        old_logps.append(lp)
        rewards.append(r_pred)
        masks.append(1-float(done))
        values.append(v)

        s = s_next
        if done:
            s = env.reset()
            if isinstance(s, tuple): s = s[0]
            done = False

    # pack up
    states   = torch.cat(states)
    actions  = torch.stack(actions)
    old_logps= torch.stack(old_logps)
    values   = np.array(values + [0.0], dtype=np.float32)

    return states, actions, old_logps, rewards, masks, values

def train_ppo():
    for it in range(1, 1001):
        s, a, lp, r, m, v = collect_batch()
        advs, rets       = compute_gae(r, v, m)
        advs = torch.tensor((advs-advs.mean())/(advs.std()+1e-8), device=DEVICE)
        rets = torch.tensor(rets, device=DEVICE)

        # PPO epochs
        N = s.size(0)
        for _ in range(PPO_EPOCHS):
            idx = np.random.permutation(N)
            for start in range(0, N, MINI_BATCH):
                mb = idx[start:start+MINI_BATCH]
                s_mb   = s[mb]
                a_mb   = a[mb]
                lp_mb  = lp[mb]
                adv_mb = advs[mb]
                ret_mb = rets[mb]

                # policy update
                dist_new = Categorical(policy(s_mb))
                lp_new   = dist_new.log_prob(a_mb)
                lp = lp.detach()            # no grad history
                ratio    = torch.exp(lp_new - lp_mb)
                cl_adv   = torch.clamp(ratio,1-CLIP_EPS,1+CLIP_EPS)*adv_mb
                loss_pi  = -torch.min(ratio*adv_mb, cl_adv).mean()

                # value update
                loss_v   = F.mse_loss(value(s_mb), ret_mb)

                opt_pi.zero_grad()
                opt_v.zero_grad()
                
                loss = loss_pi + loss_v
                loss.backward()
                
                opt_pi.step()
                opt_v.step()

        # periodic true-environment eval
        if it % 100 == 0:
            returns = []
            for _ in range(20):
                o, done, R = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset(), False, 0
                while not done:
                    _, lp = policy.act(o)
                    a     = lp.argmax().item()
                    step_out = env.step(a)
                    if len(step_out) == 5:
                        o, r, terminated, truncated, _ = step_out
                        done = terminated or truncated
                    else:
                        o, r, done, _ = step_out
                    R += r
                returns.append(R)
            print(f"[Iter {it}] True-Env Return: {np.mean(returns):.1f}±{np.std(returns):.1f}")


if __name__ == "__main__":
    # ——— Setup ———
    env     = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    rm = RewardModel(obs_dim, act_dim).to(DEVICE)
    rm.load_state_dict(torch.load(f'{CHECKPOINT_DIR}\\reward_model.pth', map_location=DEVICE))
    rm.eval()

    policy = Policy(state_size=obs_dim, action_size=act_dim).to(DEVICE)
    policy.load_state_dict(torch.load(f"{DATA_DIR}\policy_pi2.pth", map_location=DEVICE))
    value = ValueNet(obs_dim).to(DEVICE)
    value.load_state_dict(torch.load(f"{CHECKPOINT_DIR}\\value_net.pth", map_location=DEVICE))
    opt_pi = torch.optim.Adam(policy.parameters(), lr=LR_PI)
    opt_v  = torch.optim.Adam(value.parameters(),  lr=LR_V)

    train_ppo()
    torch.save(policy.state_dict(), f"{DATA_DIR}\\ppo_policy.pth")
    print(f"Saved PPO policy to {DATA_DIR}\\ppo_policy.pth")

