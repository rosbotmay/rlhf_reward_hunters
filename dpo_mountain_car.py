import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
import random
from torch.utils.data import DataLoader, Dataset


# Model definition
class PreferenceModel(nn.Module):
    def __init__(self):
        super(PreferenceModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),          # Reduced neurons to 32
            nn.ReLU(),                 # Using ReLU for simplicity
            nn.Linear(64, 32),         # Further reduction to 16
            nn.ReLU(),                 # Using ReLU for simplicity
            nn.Linear(32, 1)         # Further reduction to 16
        )

    def forward(self, x):
        return self.net(x)

# Binary Cross-Entropy Loss with Logits
criterion = nn.BCEWithLogitsLoss()

# Improved optimizer with weight decay for regularization
model = PreferenceModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# Training function
def train_model(dataset, val_dataset, epochs=100, patience=20):
    losses = []
    best_loss = float('inf')
    early_stop_count = 0

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for data in dataset:
            traj1, traj2, preferred = data['traj1'], data['traj2'], data['preferred']

            obs1 = torch.tensor([step['obs'] for step in traj1], dtype=torch.float32)
            obs2 = torch.tensor([step['obs'] for step in traj2], dtype=torch.float32)

            obs1_mean = torch.mean(obs1, dim=0)
            obs2_mean = torch.mean(obs2, dim=0)

            input_features = torch.cat([obs1_mean, obs2_mean])
            label = torch.tensor([1.0 if preferred == 1 else 0.0], dtype=torch.float32)

            optimizer.zero_grad()
            output = model(input_features)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_dataset:
                traj1, traj2, preferred = data['traj1'], data['traj2'], data['preferred']

                obs1 = torch.tensor([step['obs'] for step in traj1], dtype=torch.float32)
                obs2 = torch.tensor([step['obs'] for step in traj2], dtype=torch.float32)

                obs1_mean = torch.mean(obs1, dim=0)
                obs2_mean = torch.mean(obs2, dim=0)

                input_features = torch.cat([obs1_mean, obs2_mean])
                label = torch.tensor([1.0 if preferred == 1 else 0.0], dtype=torch.float32)

                output = model(input_features)
                val_loss += criterion(output, label).item()

        # Average losses
        avg_loss = total_loss / len(dataset)
        avg_val_loss = val_loss / len(val_dataset)
        losses.append(avg_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        """if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_count = 0  # Reset counter if loss improves
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
            print(f"✅ Improvement! Saving model with val loss: {best_loss:.4f}")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f"⛔ Early stopping triggered after {patience} epochs without improvement.")
                break"""

        scheduler.step()

    print("Training complete!")
    return losses

# Evaluation function
def evaluate_model(model, dataset, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for pair in dataset:
            # Extract and average observations from both trajectories
            obs1 = torch.tensor([step["obs"] for step in pair["traj1"]], dtype=torch.float32)
            obs2 = torch.tensor([step["obs"] for step in pair["traj2"]], dtype=torch.float32)

            obs1_mean = torch.mean(obs1, dim=0)
            obs2_mean = torch.mean(obs2, dim=0)

            # Concatenate the averaged observations to create the input
            input_features = torch.cat([obs1_mean, obs2_mean], dim=0).unsqueeze(0)
            preferred = torch.tensor([1.0 if pair['preferred'] == 1 else 0.0], dtype=torch.float32)

            # Forward pass through the model
            logits = model(input_features).squeeze()

            #print(f"Logit after model(input_features).squeeze() : {logits} \n")

            # Ensure the shapes match for the loss function
            logits = logits.unsqueeze(0)  # Convert scalar to tensor of shape [1]

            # Loss calculation
            loss = criterion(logits, preferred)
            total_loss += loss.item()

            # Prediction and accuracy calculation
            prob = torch.sigmoid(logits).item()
            prediction = 1.0 if prob > 0.5 else 0.0
            correct += (prediction == preferred.item())
            total += 1

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy




# Load and split dataset
with open('preference_dataset_100.json', 'r') as f:
    dataset = json.load(f)

# Count the occurrences of traj1 and traj2 being preferred
preferences = [item['preferred'] for item in dataset]
counts = Counter(preferences)

# Plotting the preference balance
labels = ['traj1 preferred', 'traj2 preferred']
sizes = [counts[1], counts[0]]
colors = ['skyblue', 'lightgreen']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title("Preference Balance")
plt.savefig("Generated_pairs_balance.png")


random.shuffle(dataset)
train_ratio, val_ratio = 0.7, 0.2
train_end = int(train_ratio * len(dataset))
val_end = train_end + int(val_ratio * len(dataset))
train_set, val_set, test_set = dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]


# Train the model
losses = train_model(train_set,val_set)

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b')
plt.title("DPO Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("dpo_training_loss.png")

# Evaluate the model
val_loss, val_acc = evaluate_model(model, val_set, criterion)
test_loss, test_acc = evaluate_model(model, test_set, criterion)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2%}")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")


