from policy import Policy
from config import ENV_NAME, DEVICE, DATA_DIR, CHECKPOINT_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym

# ——— Setup ———
env     = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

# ——— A simple ValueNet (critic) ———
class ValueNet(nn.Module):
    def __init__(self, state_size=4, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# 1. Collect (state, return) from the pretrained policy
def collect_mc_data(env, policy, gamma=0.99, n_episodes=50):
    data = []  # list of (state, G) pairs
    for _ in range(n_episodes):
        traj = []
        s = env.reset()
        if isinstance(s, tuple): s = s[0]
        done = False
        while not done:
            a, _ = policy.act(s)
            step = env.step(a)
            if len(step) == 5:
                s_next, r, term, trunc, _ = step
                done = term or trunc
            else:
                s_next, r, done, _ = step
            traj.append((s, r))
            s = s_next
        # back-compute returns
        G = 0
        for (s_t, r_t) in reversed(traj):
            G = r_t + gamma * G
            data.append((s_t, G))
    return data

# 2. Pretrain the ValueNet
def pretrain_value_net(value_net, optimizer, data, batch_size=64, epochs=5):
    value_net.train()
    for ep in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            states = torch.tensor([s for s, _ in batch],
                                  dtype=torch.float32, device=DEVICE)
            returns = torch.tensor([G for _, G in batch],
                                   dtype=torch.float32, device=DEVICE)

            preds = value_net(states)
            loss = F.mse_loss(preds, returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        print(f"[Pretrain Critic] Epoch {ep+1}/{epochs}  MSE={total_loss/len(data):.3f}")
    value_net.eval()


if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    policy = Policy(obs_dim, act_dim).to(DEVICE)
    policy.load_state_dict(torch.load(f'{DATA_DIR}\policy_pi2.pth', map_location=DEVICE))

    value_net = ValueNet(obs_dim).to(DEVICE)
    opt_v = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    mc_data = collect_mc_data(env, policy, gamma=0.99, n_episodes=100)
    pretrain_value_net(value_net, opt_v, mc_data, batch_size=128, epochs=20)
    torch.save(value_net.state_dict(), f"{CHECKPOINT_DIR}\\value_net.pth")
    print(f"Saved value_net to {CHECKPOINT_DIR}\\value_net.pth")
