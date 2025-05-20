import numpy as np
import torch
import gym
import torch.nn as nn
from collections import deque
from policy import Policy
import torch.optim as optim

# PLOT 1: vanilla REINFORCE
# --> with gradient estimator according to version 1 of the PG theorem
def reinforce(env, policy, optimizer, early_stop=False, n_episodes=1000, max_t=200, gamma=0.99, print_every=100, max_reward = -110):
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            if done:
                break
        # Calculate total expected reward
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Recalculate the total reward applying discounted factor
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a,b in zip(discounts, rewards)])

        # Calculate the loss
        policy_loss = []
        for log_prob in saved_log_probs:
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * R)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
        if early_stop and np.mean(scores_deque) >= max_reward:
            print('Reached the wanted reward: {:.2f} Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(max_reward, e - 100, np.mean(scores_deque)))
            break
    return scores



# PLOT 2: reward-to-go REINFORCE
# --> with gradient estimator according to version 2 of the PG theorem (not using Q-values, but reward to go)
def reinforce_rwd2go(env, policy, optimizer, early_stop=False, n_episodes=1000, max_t=200, gamma=0.99, print_every=100, max_reward = 495):
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            if done:
                break
        # Calculate total expected reward
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Recalculate the total reward applying discounted factor
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        rewards_to_go = [sum([discounts[j]*rewards[j+t] for j in range(len(rewards)-t) ]) for t in range(len(rewards))]

        # Calculate the loss
        policy_loss = []
        for i in range(len(saved_log_probs)):
            log_prob = saved_log_probs[i]
            G = rewards_to_go[i]
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * G)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
        if early_stop and np.mean(scores_deque) >= max_reward:
            print('Reached the wanted reward: {:.2f} Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(max_reward, e - 100, np.mean(scores_deque)))
            break
    return scores



def naive_baseline(state):
    # For MountainCar, state[0]=position, state[1]=velocity
    # A simple baseline: encourage higher position (closer to goal at 0.5)
    return 100 * (state[0] - (-0.5))



# PLOT 3: reward-to-go with baseline REINFORCE
# --> with gradient estimator according to version 3 of the PG theorem (not using Q-values, but reward to go)
# --> here, we consider only fixed (handcrafted) baseline functions b : S -> R; clearly, training a NN to predict V^{\pi}(s) as a baseline is also possible (and interesting!)
def reinforce_rwd2go_baseline(
    env, policy, optimizer, early_stop=False, baseline=naive_baseline,
    n_episodes=1000, max_t=200, gamma=0.99, print_every=100, max_reward=-110
):
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        saved_log_probs = []
        rewards = []
        baseline_values = []
        state, _ = env.reset()
        # Collect trajectory
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            baseline_values.append(baseline(state))
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Discounted rewards-to-go
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        rewards_to_go = [
            sum([discounts[j] * rewards[j + t] for j in range(len(rewards) - t)])
            for t in range(len(rewards))
        ]

        # Policy loss with baseline
        policy_loss = []
        for i in range(len(saved_log_probs)):
            log_prob = saved_log_probs[i]
            G_centered = rewards_to_go[i] - baseline_values[i]
            policy_loss.append(-log_prob * G_centered)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
        if early_stop and np.mean(scores_deque) >= max_reward:
            print('Reached the wanted reward: {:.2f} Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(max_reward, e - 100, np.mean(scores_deque)))
            break
    return scores
