import math
import pickle

import gym
from dqn_mountain_car import DQNAgent

def generate_trajectory(agent, env):
    state = env.reset()
    # Set epsilon to low value to make the agent exploit its learned policy
    agent.epsilon = 0.01
    done = False
    trajectory = []
    total_reward = 0
    while not done:
        action = agent.select_action(state)  # Or however the agent chooses actions
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state

    return trajectory, total_reward

if __name__ == "__main__":

    # load checkpoints as agents
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent_pi1 = DQNAgent(state_size, action_size)
    agent_pi1.load("checkpoints/dqn_mountain_car_pi1.pth") # 
    agent_pi2 = DQNAgent(state_size, action_size)
    agent_pi2.load("checkpoints/dqn_mountain_car_pi2.pth") # 

    # Generate K preference pairs
    K = 200  # Number of preference pairs to generate
    preference_pairs = []

    for _ in range(K):
        # Generate a trajectory from policy pi1 and pi2
        trajectory_pi1, reward_pi1 = generate_trajectory(agent_pi1, env)
        trajectory_pi2, reward_pi2 = generate_trajectory(agent_pi2, env)

        # Compute the probability that pi1's trajectory is preferred
        prob_pi1 = math.exp(reward_pi1) / (math.exp(reward_pi1) + math.exp(reward_pi2))
        preference_pairs.append((trajectory_pi1, trajectory_pi2, prob_pi1))
    
    print([p for pi1,pi2,p in preference_pairs])

    # Save the preference pairs
    with open(f"data/preference_pairs_{K}.pkl", "wb") as f:
        pickle.dump(preference_pairs, f)
