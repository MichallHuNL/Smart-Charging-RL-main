from torch import nn
import torch as th
import torch.nn.functional as F

class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(CentralizedCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim * num_agents, 128)
        self.fc2 = nn.Linear(128 + 1 * num_agents, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = F.relu(self.fc1(states))
        x = th.cat([x, actions], dim=-1)
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value