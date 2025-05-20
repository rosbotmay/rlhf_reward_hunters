import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from config import DEVICE

torch.manual_seed(0)

STATE_SIZE = 2
ACTION_SIZE = 3

class Policy(nn.Module):
    """
    Neural network-based policy for reinforcement learning.

    Args:
        state_size (int): Dimension of the input state.
        action_size (int): Number of possible actions.
        hidden_size (int): Number of units in the hidden layer.

    Methods:
        forward(state): Computes action probabilities from input state.
        act(state): Samples an action and returns its log probability.
    """
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_size=32):
        """
        Initializes the Policy network.

        Args:
            state_size (int): Dimension of the input state.
            action_size (int): Number of possible actions.
            hidden_size (int): Number of units in the hidden layer.
        """
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Forward pass to compute action probabilities.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Softmax probabilities over actions.
        """
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        # we just consider 1 dimensional probability of action
        return F.softmax(x, dim=1)

    def act(self, state):
        """
        Selects an action based on the current policy.

        Args:
            state (np.ndarray): The current state.

        Returns:
            tuple: (action (int), log probability of the action (torch.Tensor))
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE) 
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)
    

class RandomPolicy:
    """
    Policy that selects actions randomly from the action space.

    Args:
        action_space: The action space with a sample() method.

    Methods:
        act(_): Returns a random action and None for log probability.
    """
    def __init__(self, action_space):
        """
        Initializes the RandomPolicy.

        Args:
            action_space: The action space with a sample() method.
        """
        self.action_space = action_space

    def act(self, _):
        """
        Selects a random action.

        Args:
            _ : Ignored input.

        Returns:
            tuple: (random action, None)
        """
        return self.action_space.sample(), None