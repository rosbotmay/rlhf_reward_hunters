import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DPOModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DPOModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single value representing the preference score

    def forward(self, features_pi1, features_pi2):
        combined = torch.cat((features_pi1, features_pi2), dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out # Output a single value (preference score)

def get_features(trajectory):
    """
    Extracts simple features from a trajectory.
    For example, using the sum and average of the rewards in the trajectory.
    """
    rewards = [step[2] for step in trajectory]  # Step[2] is the reward in the trajectory
    sum_rewards = np.sum(rewards)
    avg_rewards = np.mean(rewards) if len(rewards) > 0 else 0
    trajectory_length = len(rewards)

    # Combine the features into a single feature vector
    features = np.array([sum_rewards, avg_rewards, trajectory_length], dtype=np.float32)

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension for PyTorch

# DPO training class
class DPO:
    def __init__(self, model, preference_pairs, lr=1e-5):
        self.model = model
        self.preference_pairs = preference_pairs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for trajectory_pi1, trajectory_pi2, prob_pi1 in self.preference_pairs:
                # Extract features for the trajectories (shape [1, 3] for each)
                features_pi1 = get_features(trajectory_pi1)  # Shape: (1, 3)
                features_pi2 = get_features(trajectory_pi2)  # Shape: (1, 3)

                # Compute preference probability prediction (output shape: [1, 1])
                pred_prob = self.model(features_pi1, features_pi2)  # Forward pass

                # Calculate the target (probability that trajectory 1 is preferred)
                target = torch.tensor(prob_pi1, dtype=torch.float32).unsqueeze(0)  # Shape: (1,)

                # Ensure the target tensor has the same shape as the prediction (1, 1)
                target = target.unsqueeze(1)  # Shape: (1, 1)

                # Compute loss
                loss = self.loss_fn(pred_prob, target)
                total_loss += loss.item()

                # Update the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(self.preference_pairs)}")


