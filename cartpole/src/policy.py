import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from config import DEVICE

torch.manual_seed(0)

class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        # we just consider 1 dimensional probability of action
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE) 
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)
    

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, _):
        return self.action_space.sample(), None