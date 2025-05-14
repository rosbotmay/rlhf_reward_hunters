import torch
import numpy as np
from dqn_pendulum import DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_agent(checkpoint_path: str, state_dim: int, action_space: np.ndarray) -> DQNAgent:
    agent = DQNAgent(state_dim=state_dim, action_space=action_space)
    # Load the checkpoint with weights_only=False to include all data
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.state_normalizer.mean = checkpoint['state_normalizer_mean']
    agent.state_normalizer.std = checkpoint['state_normalizer_std']
    agent.state_normalizer.n = checkpoint['state_normalizer_n']
    agent.epsilon = 0  # Greedy policy
    return agent


def generate_trajectory(agent, env, max_steps=200):
    action_space = np.linspace(-2.0, 2.0, 50)  # Pendulum-v1: [-2, 2]
    state, _ = env.reset()
    total_reward = 0
    trajectory = []
    for _ in range(max_steps):
        state_normalized = agent.state_normalizer.normalize(state)
        action_idx = agent.select_action(state_normalized)
        action = action_space[action_idx]
        next_state, reward, done, truncated, _ = env.step([action])
        theta = np.arctan2(next_state[1], next_state[0])
        reward += 0.5 * (1 - abs(theta) / np.pi)  # Same shaping as training
        trajectory.append((state, action_idx, np.float32(action), np.float32(reward)))
        total_reward += reward
        state = next_state
        if done or truncated:
            break
    return trajectory, np.float32(total_reward)