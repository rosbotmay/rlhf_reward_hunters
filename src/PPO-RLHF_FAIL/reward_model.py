import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pickle, random
from config import DATA_DIR, CHECKPOINT_DIR
import gym
from config import ENV_NAME, DEVICE

class RewardModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = F.relu(self.fc1(x))
        return self.out(F.relu(self.fc2(h))).squeeze(-1)


def train_reward_model(obs_dim, act_dim, epochs=10, batch_size=32, lr=1e-4, split=0.2):
    # 1. Load and split
    with open(f"{DATA_DIR}\prefs_expert_vs_subexpert.pkl",'rb') as f: prefs = pickle.load(f)
    random.shuffle(prefs)
    n_val = int(len(prefs)*split)
    train, val = prefs[n_val:], prefs[:n_val]
    train_loader = DataLoader(train,  batch_size=batch_size, shuffle=True,  collate_fn=lambda x:x)
    val_loader   = DataLoader(val,    batch_size=batch_size, shuffle=False, collate_fn=lambda x:x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rm = RewardModel(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(rm.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        # train
        rm.train()
        total_loss = 0
        for batch in train_loader:
            losses = []
            for p in batch:
                # sum over tau1
                R1 = torch.zeros(1, device=device)
                for s,a,_ in p['tau1']:
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    a_t = (torch.zeros(act_dim,device=device)
                           .scatter_(0, torch.tensor(a,device=device), 1.)
                          ).unsqueeze(0)
                    R1 += rm(s_t, a_t)
                # sum over tau2
                R2 = torch.zeros(1, device=device)
                for s,a,_ in p['tau2']:
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    a_t = (torch.zeros(act_dim,device=device)
                           .scatter_(0, torch.tensor(a,device=device), 1.)
                          ).unsqueeze(0)
                    R2 += rm(s_t, a_t)

                logit = R1 - R2
                target = torch.tensor(p.get('p_tau1_pref', p['label']),
                                      dtype=torch.float32, device=device)
                losses.append(F.binary_cross_entropy_with_logits(logit.squeeze(), target))
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # validate
        rm.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                for p in batch:
                    # compute predicted cumulative reward for each trajectory
                    R1_hat = torch.zeros(1, device=device)
                    for s,a,_ in p['tau1']:
                        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                        a_ohe = torch.zeros(act_dim, device=device)
                        a_ohe[a] = 1.0
                        R1_hat += rm(s_t, a_ohe.unsqueeze(0))

                    R2_hat = torch.zeros(1, device=device)
                    for s,a,_ in p['tau2']:
                        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                        a_ohe = torch.zeros(act_dim, device=device)
                        a_ohe[a] = 1.0
                        R2_hat += rm(s_t, a_ohe.unsqueeze(0))

                    # preference logit and target
                    logit = (R1_hat - R2_hat).squeeze()
                    target = p.get('p_tau1_pref', p['label'])  # soft or hard label
                    val_loss += F.binary_cross_entropy_with_logits(
                        logit, torch.tensor(target, device=device)
                    ).item()

                    # hard‐label accuracy
                    pred = (torch.sigmoid(logit) > 0.5).item()
                    true = (target > 0.5)
                    correct += (pred == true)
                    total   += 1

        val_loss /= total
        accuracy = correct / total * 100
        print(f"Reward Model held-out BCE loss: {val_loss:.4f}, accuracy: {accuracy:.1f}% on {total} pairs")

    # save model
    torch.save(rm.state_dict(), f"{CHECKPOINT_DIR}/reward_model.pth")
    return rm.to('cpu')


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    env.reset(seed=0)
    obs_dim = env.observation_space.shape[0]

    # for discrete envs, one-hot encode actions
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]

    train_reward_model(obs_dim, act_dim, epochs=10, batch_size=32, lr=1e-4)