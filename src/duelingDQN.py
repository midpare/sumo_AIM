import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, ego_dim, nbr_dim, n_actions, hid=256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(nbr_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU()
        )
        self.trunk = nn.Sequential(
            nn.Linear(ego_dim + 64, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU()
        )
        self.value = nn.Sequential(nn.Linear(hid, 64), nn.SiLU(), nn.Linear(64, 1))
        self.adv   = nn.Sequential(nn.Linear(hid, 64), nn.SiLU(), nn.Linear(64, n_actions))

    def forward(self, ego, nbrs):  
        # ego: [B, ego_dim], nbrs: [B, n_nbr, nbr_dim]
        B = ego.size(0)
        phi_n = self.phi(nbrs.reshape(-1, nbrs.size(-1))).reshape(B, nbrs.size(1), -1)
        pooled = phi_n.sum(dim=1)            
        h = self.trunk(torch.cat([ego, pooled], dim=1))
    
        V = self.value(h)
        A = self.adv(h)
        Q = V + (A - A.mean(dim=1, keepdim=True))

        return Q