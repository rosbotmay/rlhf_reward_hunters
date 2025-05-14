import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt

class PPOPolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class RewardModel(nn.Module):
    def __init__(self, input_size):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        feature_vector = torch.cat((state_tensor, action_tensor), dim=-1)
        x = torch.relu(self.fc1(feature_vector))
        x = torch.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward


class PPO_RHLF:
    def __init__(self, policy, reward_model, env, gamma=0.99, epsilon=0.2, learning_rate=1e-4, batch_size=64, epochs=10, lambda_=0.95):
        self.policy = policy
        self.reward_model = reward_model
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_model_optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs

    def select_action(self, state):
        action_probs = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_gae(self, trajectory, values):
        rewards = [step[2] for step in trajectory]  # Extract rewards
        dones = [step[4] for step in trajectory]  # Extract done flags
        advantages = []
        last_gae_lam = 0
        next_value = 0  # You can use the terminal state value here

        for t in range(len(trajectory) - 1, -1, -1):  # Traverse the trajectory in reverse
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae_lam = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_gae_lam
            advantages.insert(0, last_gae_lam)
            next_value = values[t]

        return advantages

    def normalize_rewards(self, rewards):
        """
        Normalize rewards within each trajectory to have zero mean and unit variance.
        :param rewards: List of rewards
        :return: Normalized rewards
        """
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if np.std(rewards) > 0 else 1  # Prevent division by zero
        normalized_rewards = (rewards - mean_reward) / (std_reward +1e-8)
        return normalized_rewards

    def compute_rewards(self, trajectory):
        rewards = [step[2] for step in trajectory]
        normalized_rewards = self.normalize_rewards(rewards)
        total_reward = np.sum(normalized_rewards)
        mean_reward = np.mean(normalized_rewards)
        return total_reward, mean_reward

    def train_reward_model(self, trajectory):
        for state, action, reward, _, _ in trajectory:
            predicted_reward = self.reward_model(state, action)
            loss = F.mse_loss(predicted_reward, torch.tensor([reward], dtype=torch.float32).unsqueeze(0))
            #print(f"reward model loss : {loss}")
            
            self.reward_model_optimizer.zero_grad()
            loss.backward()
            self.reward_model_optimizer.step()

    def train(self, preference_pairs):
        self.losses = []

        for epoch in range(self.epochs):
            total_loss = 0

            for trajectory_pi1, trajectory_pi2, prob_pi1 in preference_pairs:
                # Train the reward model
                if epoch % 5 == 0:
                    self.train_reward_model(trajectory_pi1)
                    self.train_reward_model(trajectory_pi2)

                # Get the predicted values for each state in the trajectory
                values_pi1 = [self.reward_model(state, action).item() for state, action, _, _, _ in trajectory_pi1]
                values_pi2 = [self.reward_model(state, action).item() for state, action, _, _, _ in trajectory_pi2]

                # Compute the advantages using GAE
                advantages_pi1 = self.compute_gae(trajectory_pi1, values_pi1)
                advantages_pi2 = self.compute_gae(trajectory_pi2, values_pi2)

                # Calculate the total reward for both trajectories
                total_reward_pi1, _ = self.compute_rewards(trajectory_pi1)
                total_reward_pi2, _ = self.compute_rewards(trajectory_pi2)

                # Compute the advantage
                #advantage = prob_pi1 * total_reward_pi1 + (1 - prob_pi1) * total_reward_pi2

                for t in range(len(trajectory_pi1)):
                    state, action, reward, next_state, done = trajectory_pi1[t]
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    action, log_prob = self.select_action(state_tensor)

                    # Compute PPO loss
                    advantage = advantages_pi1[t]
                    ratio = torch.exp(log_prob)
                    clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
                    total_loss += loss.mean()

            self.losses.append(total_loss / len(preference_pairs))
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(preference_pairs)}")

            # Backpropagation and optimization for PPO policy
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.plot_loss()

        return self.losses

    def plot_loss(self):
        losses_to_plot = [loss.detach().cpu().numpy() for loss in self.losses]
        plt.plot(losses_to_plot)
        plt.title('PPO-RHLF Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f"results/ppo_rlhf_training.png")
        plt.close()
