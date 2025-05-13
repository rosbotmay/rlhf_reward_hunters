import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

torch.autograd.set_detect_anomaly(True)

# Define the PPO Policy Network
class PPOPolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs

# Dummy reward model for illustration
class RewardModel(nn.Module):
    def __init__(self, input_size):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        # Ensure both tensors have the same number of dimensions
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Shape (1, 2)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # Shape (1, 1)

        # Concatenate along the feature dimension
        feature_vector = torch.cat((state_tensor, action_tensor), dim=-1)  # Shape (1, 3)

        # Forward pass through the network
        x = torch.relu(self.fc1(feature_vector))
        x = torch.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward





# PPO-RHLF Implementation
class PPO_RHLF:
    def __init__(self, policy, reward_model, env, gamma=0.99, epsilon=0.2, learning_rate=1e-4, batch_size=64, epochs=10):
        self.policy = policy
        self.reward_model = reward_model
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs

    def select_action(self, state):
        action_probs = self.policy(state)  # Get action probabilities from the policy
        dist = torch.distributions.Categorical(action_probs)  # Use a categorical distribution
        action = dist.sample()  # Sample an action
        return action.item(), dist.log_prob(action)  # Return action and its log probability

    def compute_rewards(self, trajectory):
        rewards = [step[2] for step in trajectory]  # Step[2] is the reward in the trajectory
        return np.sum(rewards), np.mean(rewards)  # Total and average reward

    def train(self, preference_pairs):
        # Collect trajectories from the current policy and compute reward predictions using the reward model
        trajectories = []
        for _ in range(self.batch_size):
            state = self.env.reset()
            done = False
            trajectory = []
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, log_prob = self.select_action(state_tensor)  # Call the select_action method here
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state, action, reward, next_state, done, log_prob))
                state = next_state
            trajectories.append(trajectory)

        # Now train PPO using the reward model for the reward signals
        for i in range(self.epochs):
            total_loss = 0
            losses = []  # List to store individual losses for the batch
            for k, trajectory in enumerate(trajectories):
                trajectory_rewards = [step[2] for step in trajectory]
                total_reward, _ = self.compute_rewards(trajectory)
                
                for t in range(len(trajectory)):
                    state, action, reward, next_state, done, log_prob = trajectory[t]

                    # Use the reward model to get the reward for this state-action pair
                    reward_model_output = self.reward_model(state, action)

                    # Define the Advantage
                    advantage = reward_model_output - total_reward

                    # Calculate the loss using the clipped objective function of PPO
                    ratio = torch.exp(log_prob)
                    clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

                    losses.append(loss.mean())  # Store the mean of the loss for this trajectory

            # Calculate the total loss for the epoch
            total_loss = torch.stack(losses).mean()  # Mean of all losses in the batch

            # Print the loss for this epoch
            print(f"Epoch {i+1} --------- loss: {total_loss.item()}")

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=True)  # Retaining the computation graph for future passes
            self.optimizer.step()







