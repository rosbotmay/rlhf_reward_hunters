import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import os
import csv

# Set environment variable for CuBLAS to ensure deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# State Normalizer (same as PPO)
class StateNormalizer:
    def __init__(self, state_dim):
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.n = 0

    def update(self, state):
        self.n += 1
        old_mean = self.mean.copy()
        self.mean = self.mean + (state - self.mean) / self.n
        self.std = np.sqrt(self.std**2 + (state - old_mean) * (state - self.mean))

    def normalize(self, state):
        return (state - self.mean) / (self.std + 1e-8)

# DQN Agent with Improvements
class DQNAgent:
    def __init__(self, state_dim, action_space, lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99975, batch_size=64, memory_size=100000):
        self.state_dim = state_dim
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Neural networks
        self.q_network = QNetwork(state_dim, self.n_actions).to(device)
        self.target_network = QNetwork(state_dim, self.n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.memory = deque(maxlen=memory_size)

        # State normalizer
        self.state_normalizer = StateNormalizer(state_dim)

        # Track best reward
        self.best_reward = float('-inf')

    def select_action(self, state):
        state = self.state_normalizer.normalize(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array([self.state_normalizer.normalize(s) for s in states])).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        rewards = torch.clamp(rewards / 16.0, -1.0, 0.0)  # Normalize rewards like PPO
        next_states = torch.FloatTensor(np.array([self.state_normalizer.normalize(s) for s in next_states])).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Compute Q-values using Double DQN
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss and update
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, episode, checkpoint_dir="checkpoints_dqn_5000", filename_prefix="dqn_checkpoint"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'episode': episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_normalizer_mean': self.state_normalizer.mean,
            'state_normalizer_std': self.state_normalizer.std,
            'state_normalizer_n': self.state_normalizer.n,
            'best_reward': self.best_reward,
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"{filename_prefix}_episode_{episode}.pt"))
        if episode == 'best':
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"{filename_prefix}_best.pt"))

# Training Loop
def train_dqn(checkpoint_freq=100, window_size=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]

    # Finer action discretization (50 bins)
    action_space = np.linspace(env.action_space.low[0], env.action_space.high[0], 50)
    agent = DQNAgent(state_dim, action_space)

    num_episodes = 5000  # Increased training duration
    max_steps = 200
    target_update_freq = 50  # Less frequent updates

    print(f"Training on device: {device}, Seed: {seed}")

    episode_rewards = []

    # Initialize CSV file for saving rewards
    reward_file = "dqn_rewards_5000.csv"
    with open(reward_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0

        for step in range(max_steps):
            agent.state_normalizer.update(state)
            action_idx = agent.select_action(state)
            action = action_space[action_idx]
            next_state, reward, done, truncated, _ = env.step([action])
            done = done or truncated

            # Reward shaping: Add bonus for small angles
            theta = np.arctan2(next_state[1], next_state[0])
            angle_bonus = 1.0 - abs(theta) / np.pi
            reward += 0.5 * angle_bonus

            agent.store_transition(state, action_idx, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.update_epsilon()
        if episode % target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(episode_reward)

        # Save episode reward to CSV
        with open(reward_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, episode_reward])

        # Log new best reward
        if episode_reward > agent.best_reward:
            agent.best_reward = episode_reward
            agent.save_checkpoint(f'best_at_{episode + 1}')
            tqdm.write(f"New best reward: {agent.best_reward:.2f}, saved best checkpoint")

        # Log average reward every 100 episodes and save checkpoint
        if (episode + 1) % checkpoint_freq == 0:
            agent.save_checkpoint(episode + 1)
            if len(episode_rewards) >= window_size:
                avg_reward = np.mean(episode_rewards[-window_size:])
                tqdm.write(f"Episode {episode + 1}, Avg Reward (last {window_size}): {avg_reward:.2f}, Saved checkpoint")
            else:
                tqdm.write(f"Episode {episode + 1}, Saved checkpoint")

    env.close()
    return agent, action_space

if __name__ == "__main__":
    agent, action_space = train_dqn(seed=42)