import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Define the Q-network (a simple feedforward neural network)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)  # Output shape should be (batch_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x).squeeze(1)
        """print(f"self.fc3(x) : {out.shape}")"""
        return out  # This should return shape (batch_size, num_actions)

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99999, learning_rate=1e-3, batch_size=64, memory_size=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Memory
        self.memory = deque(maxlen=memory_size)

        # Initialize Q-network and target network
        self.qnetwork = QNetwork(state_size, action_size)
        self.target_qnetwork = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)

        # Sync target network with the main Q-network
        self.update_target_network()

    def update_target_network(self):
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.qnetwork(state)
        return torch.argmax(q_values).item()  # Exploit

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)

        # Convert dones to float for mathematical operations
        dones = dones.float()

        # Q-values for current states (Shape: [batch_size, num_actions])
        q_values = self.qnetwork(states)  # Shape should be (batch_size, num_actions)

        # Check the shape before indexing
        """print(f"q_values shape: {q_values.shape}")  # Should be (batch_size, num_actions)"""
        
        # If there is an extra dimension (batch_size, 1, num_actions), remove it
        #q_values = q_values.squeeze(1)  # Now shape should be (batch_size, num_actions)

        """print(f"q_values squueezed shape: {q_values.shape}")  # Should be (batch_size, num_actions)"""

        # Select the Q-values for the chosen actions
        q_values = q_values[torch.arange(q_values.size(0)), actions]  # Shape: (batch_size,)

        # Q-values for next states
        next_q_values = self.target_qnetwork(next_states).max(1)[0]  # Shape: (batch_size, 1)
        
        """print(f"next_q_values shape: {next_q_values.shape}")
        print(f"dones shape: {dones.shape}")"""

        # Now we can safely perform arithmetic with dones
        target = rewards.unsqueeze(1) + (self.gamma * next_q_values) * (1 - dones)

        # Loss and optimization
        loss = nn.MSELoss()(q_values, target.squeeze(1))  # Make sure target is also (batch_size,)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def select_action(self, state):
        """
        Select an action based on epsilon-greedy strategy.
        - With probability epsilon, choose a random action (exploration).
        - With probability (1 - epsilon), choose the action with the highest Q-value (exploitation).
        """
        # Convert state to tensor if it's not already
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Adding batch dimension

        if torch.rand(1).item() < self.epsilon:  # Exploration
            action = torch.randint(0, self.action_size, (1,)).item()  # Random action
        else:  # Exploitation (select the best action based on Q-values)
            with torch.no_grad():  # No need to compute gradients for inference
                q_values = self.qnetwork(state)  # Get Q-values for the current state
                action = torch.argmax(q_values).item()  # Choose the action with the highest Q-value

        return action

    def load(self, name):
        self.qnetwork.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.qnetwork.state_dict(), name)

def DQN_training(agent,env,episodes):

    rewards = []
    saved_pi1 = False
    saved_pi2 = False
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward
            

            if done:
                agent.update_target_network()
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon}")
        rewards.append(total_reward)
        # Save the model after every 50 episodes

        if e % 500 == 0:
            agent.save(f'checkpoints/dqn_mountain_car_{e}.pth')

        if e > 3000 and total_reward > -100 and not saved_pi1:
            agent.save(f"checkpoints/dqn_mountain_car_pi1.pth")
            saved_pi1 = True
        elif e >= 1500 and -130 <= total_reward <= -110 and not saved_pi2:
            agent.save(f"checkpoints/dqn_mountain_car_pi2.pth")
            saved_pi2 = True
        
       
    # Save the final model
    agent.save('checkpoints/dqn_mountain_car_final.pth')

    return rewards

def test_agent(agent, env, num_episodes=1):
    total_rewards = []
    
    # Set epsilon to low value to make the agent exploit its learned policy
    agent.epsilon = 0.01
    
    for ep in range(num_episodes):
        state = env.reset()  # Reset environment
        done = False
        total_reward = 0
        
        # Track the state and action sequence for plotting
        states = []
        actions = []
        rewards = []
        
        while not done:
            # Render the environment (use for visualization)
            env.render()
            
            # Select action using the trained policy (no exploration)
            action = agent.select_action(state)  # This should use the trained policy
            
            # Take the action and get the next state and reward
            next_state, reward, done, info = env.step(action)
            
            # Accumulate total reward
            total_reward += reward
            
            # Save data for plotting
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Update state for the next step
            state = next_state
        
        # Store total reward for this episode
        total_rewards.append(total_reward)
        
        # Plot the agent's performance during the episode
        print(f"Episode {ep + 1}: Total Reward = {total_reward}")
        
        # Optionally: Plot trajectory (position vs. time)
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(states)), [state[0] for state in states], label="Agent Position")
        plt.title(f"Agent Trajectory for Episode {ep + 1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Position")
        # Save the plot as a PNG file
        plt.savefig(f"results/agent_trajectory_episode_{ep+1}.png")
    

    return total_rewards

def plot_training_evlolution(rewards):
    # Plot the rewards per episode
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode over Time')
    # Save the plot as a PNG file
    plt.savefig("results/training_evolution.png")


# Main code to train the agent
if __name__ == "__main__":
    # Set up environment
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    episodes = 4000
    
    rewards = DQN_training(agent,env,episodes)
    plot_training_evlolution(rewards)

    """tst_agent = DQNAgent(state_size, action_size)
    tst_agent.load("checkpoints/dqn_mountain_car_pi2_.pth")"""
    #test_agent(agent,env,3)
    



