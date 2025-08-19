import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.fc_shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.fc_value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        x = self.fc_shared(x)
        value = self.fc_value(x)
        adv = self.fc_advantage(x)
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q
