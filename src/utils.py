import gym
import numpy as np
import random
from math import exp
import torch

# Set the random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # for CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Generate a trajectory using the policy
def generate_trajectory(policy, env, max_t=1000):
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
    env = gym.make(env_name)
    bucket = []
    for _ in range(K):
        traj, R = generate_trajectory(policy, env)
        bucket.append({
            "trajectory": traj,
            "total_reward": R
        })
    return bucket

# Sample a pair of trajectories from two buckets and compute the preference label
def make_pref_dataset(bucket_a, bucket_b, seed=None):
    """
    Returns a list of dicts, each with:
      - 'tau1', 'tau2' : the two trajectories
      - 'p_tau1_pref'  : BT probability
      - 'label'       : sampled hard label (1 if tau1 preferred)
      - 'R1', 'R2'    : their total rewards
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


def validate_prefs(pref_path, sample_size=1000):
    """
    Load a preference dataset and report basic statistics to check for anomalies.
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
    if isinstance(action_space, gym.spaces.Discrete):
        a_t = torch.zeros(act_dim, device=device)
        a_t[a] = 1.0
    else:
        a_t = torch.tensor(a, dtype=torch.float32, device=device)
    return a_t

# Custom collate to handle variable-length trajectories
def collate_identity(batch):
    return batch
