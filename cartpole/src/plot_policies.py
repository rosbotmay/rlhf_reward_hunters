import gym
import torch
import matplotlib.pyplot as plt
from policy import Policy
from config import DEVICE, DATA_DIR, ENV_NAME

def load_policy(checkpoint_fname):
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]

    pi = Policy(state_size=obs_dim, action_size=act_dim).to(DEVICE)
    pi.load_state_dict(torch.load(f"{DATA_DIR}/{checkpoint_fname}", map_location=DEVICE))
    pi.eval()
    return pi

def collect_angles(pi, max_steps=500):
    env = gym.make(ENV_NAME)
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    angles = []
    for t in range(max_steps):
        # cartPole’s state is [x, x_dot, theta, theta_dot]
        theta = obs[2]
        angles.append(theta)

        # act under policy
        action, _ = pi.act(obs)
        step = env.step(action)

        if len(step) == 5:
            obs, reward, done, trunc, _ = step
            done = done or trunc
        else:
            obs, reward, done, _ = step
        if done:
            break
    return angles

# load the two checkpoints:
pi1 = load_policy("policy_pi1.pth")    # expert
pi2 = load_policy("policy_pi2.pth")    # sub-expert

# collect one rollout each
angles1 = collect_angles(pi1, max_steps=500)
angles2 = collect_angles(pi2, max_steps=500)

# plot
plt.figure(figsize=(8,4))
plt.plot(angles1, label="Expert π₁")
plt.plot(angles2, label="Sub-expert π₂", alpha=0.7)
plt.axhline(0, color="k", linewidth=0.5, linestyle="--")
plt.xlabel("Timestep")
plt.ylabel("Pole angle θ (rad)")
plt.title("CartPole rollouts: expert vs sub-expert")
plt.legend()
plt.tight_layout()
plt.show()
