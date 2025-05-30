import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from config import ENV_NAME,MODELS_DIR,DATASIZES2

class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class PreferenceDataset(Dataset):
    def __init__(self, preference_pairs):
        self.data = []
        for traj1, traj2, prob in preference_pairs:
            # Convert trajectories to fixed-size vectors
            t1 = torch.tensor([list(s) + [a] for (s, a, _, _, _) in traj1], dtype=torch.float32)
            t2 = torch.tensor([list(s) + [a] for (s, a, _, _, _) in traj2], dtype=torch.float32)

            # Average over time (each is now shape [3])
            t1 = t1.mean(dim=0)
            t2 = t2.mean(dim=0)

            label = 1 if prob > 0.5 else 0  # Binary label
            self.data.append((t1, t2, torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




def train_reward_model(model, preference_pairs, epochs=100, batch_size=32, lr=1e-3):
    dataset = PreferenceDataset(preference_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for traj1, traj2, label in dataloader:
            r1 = model(traj1)  # shape [B]
            r2 = model(traj2)  # shape [B]
            logits = r1 - r2   # preference: traj1 > traj2 → positive

            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch}: loss = {total_loss:.4f}")

def evaluate_reward_model(model, preference_pairs,tag="preference"):
    model.eval()
    correct = 0
    total = len(preference_pairs)

    with torch.no_grad():
        for traj1, traj2, prob in preference_pairs:
            t1 = torch.tensor([list(s) + [a] for (s, a, _, _, _) in traj1], dtype=torch.float32).mean(dim=0)
            t2 = torch.tensor([list(s) + [a] for (s, a, _, _, _) in traj2], dtype=torch.float32).mean(dim=0)

            r1 = model(t1)
            r2 = model(t2)

            pred = 1 if r1 > r2 else 0
            label = 1 if prob > 0.5 else 0

            if pred == label:
                correct += 1

    accuracy = correct / total
    print(f"Reward model accuracy on {total} {tag} pairs: {accuracy:.2%}")

if __name__ == "__main__":

    setting_by_size =  [(500,3e-4),(600,1e-4),(700,1e-4),(1000,1e-4)]

    with open("data/preference_pairs_500_original.pkl", "rb") as f:
        evaluation_pairs = pickle.load(f)

    with open(f"data/preference_pairs_2000_original.pkl", "rb") as f:
            prefs = pickle.load(f)

    for i,size in enumerate(DATASIZES2):

        learning_rate = setting_by_size[i][1]
        preference_pairs = prefs[:size]

        model = RewardModel(input_dim=3)  # state_dim (2) + action_dim (1) = 3
        train_reward_model(model, preference_pairs, epochs=1000,lr=learning_rate)
        evaluate_reward_model(model, preference_pairs)
        evaluate_reward_model(model,evaluation_pairs,tag="evaluation")

        # Optional: save the model
        torch.save(model.state_dict(), f"{MODELS_DIR}/reward_model_{size}.pth")
