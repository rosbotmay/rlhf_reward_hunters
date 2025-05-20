import gym
import numpy as np
import random
from math import exp
import torch

# Generate a trajectory using the policy
def generate_trajectory(policy, env, max_t=200):
    """
    Runs a single episode in the given environment using the provided policy.

    Args:
        policy: The policy object with an act() method.
        env: The environment to interact with.
        max_t: Maximum number of timesteps per episode.

    Returns:
        trajectory: List of (state, action, reward) tuples.
        total_reward: Sum of rewards obtained in the episode.
    """
    state, _ = env.reset()
    trajectory = []
    total_reward = 0
    for _ in range(max_t):
        action, _ = policy.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append((state, action, reward))
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    return trajectory, total_reward

# Collect K trajectories and their total rewards
def collect_bucket(policy, env_name, K):
    """
    Collects K trajectories and their total rewards using the given policy and environment.

    Args:
        policy: The policy object with an act() method.
        env_name: Name of the gym environment.
        K: Number of trajectories to collect.

    Returns:
        bucket: List of dicts with 'trajectory' and 'total_reward' keys.
    """
    env = gym.make(env_name)
    bucket = []
    for _ in range(K):
        traj, R = generate_trajectory(policy, env)
        bucket.append({
            "trajectory": traj,
            "total_reward": R
        })
    return bucket

# Sample a pair of trajectories from two buckets and compute the preference label using Bradley-Terry model
# and the total rewards of each trajectory.
def make_pref_dataset(bucket_a, bucket_b, seed=None):
    """
    Creates a preference dataset by pairing trajectories from two buckets and assigning preference labels.

    Args:
        bucket_a: List of trajectory dicts (from collect_bucket).
        bucket_b: List of trajectory dicts (from collect_bucket).
        seed: Optional random seed for reproducibility.

    Returns:
        pairs: List of dicts with keys:
            - 'tau1', 'tau2': the two trajectories
            - 'p_tau1_pref': Bradley-Terry probability tau1 is preferred
            - 'label': 1 if tau1 is preferred, else 0
            - 'R1', 'R2': total rewards of tau1 and tau2
    """
    if seed is not None:
        random.seed(seed)
    pairs = []
    for a, b in zip(bucket_a, bucket_b):
        R1, R2 = a["total_reward"], b["total_reward"]
        p1 = exp(R1) / (exp(R1) + exp(R2))
        label = 1 if random.random() < p1 else 0
        pairs.append({
            "tau1":        a["trajectory"],
            "tau2":        b["trajectory"],
            "p_tau1_pref": p1,
            "label":       label,
            "R1":          R1,
            "R2":          R2,
        })
    return pairs


def validate_prefs(pref_path, sample_size=200):
    """
    Loads a preference dataset and prints basic statistics for validation.

    Args:
        pref_path: Path to the pickled preference dataset.
        sample_size: Number of samples to use for statistics (if dataset is large).

    Prints:
        Dataset statistics including label fraction, return difference, and BT probability.
    """
    import pickle, random
    with open(pref_path, 'rb') as f:
        prefs = pickle.load(f)
    # Sample subset if large
    if len(prefs) > sample_size:
        prefs = random.sample(prefs, sample_size)

    # Collect statistics
    R_diffs = [p['R1'] - p['R2'] for p in prefs]
    labels = [p['label'] for p in prefs]
    p_prefs = [p['p_tau1_pref'] for p in prefs]

    import numpy as np
    print(f"Dataset: {pref_path}")
    print(f"Total pairs: {len(labels)}")
    print(f"Label=1 fraction (tau1 preferred): {np.mean(labels):.3f}")
    print(f"Return diff: mean={np.mean(R_diffs):.3f}, std={np.std(R_diffs):.3f}")
    print(f"BT probability: mean={np.mean(p_prefs):.3f}, std={np.std(p_prefs):.3f}")


def action_to_tensor(a, action_space, act_dim, device):
    """
    Converts an action to a PyTorch tensor suitable for neural network input.

    Args:
        a: The action (int for discrete, array-like for continuous).
        action_space: The gym action space.
        act_dim: Action dimension (for discrete spaces).
        device: PyTorch device.

    Returns:
        a_t: Action as a PyTorch tensor.
    """
    if isinstance(action_space, gym.spaces.Discrete):
        a_t = torch.zeros(act_dim, device=device)
        a_t[a] = 1.0
    else:
        a_t = torch.tensor(a, dtype=torch.float32, device=device)
    return a_t

# Custom collate to handle variable-length trajectories
def collate_identity(batch):
    """
    Identity collate function for DataLoader to handle variable-length trajectories.

    Args:
        batch: List of samples.

    Returns:
        The batch unchanged.
    """
    return batch
