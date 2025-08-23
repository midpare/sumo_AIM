import torch, random
import numpy as np
from enum import Enum

from duelingDQN import DuelingDQN
from replayBuffer import PrioritizedReplayBuffer

class AgentType(Enum):
    LEFT = "left"
    STRAIGHT = "straight" 
    RIGHT = "right"
    
class D3QNAgent:
    def __init__(self, name, ego_dim, nbr_dim, n_nbr, n_actions, mean_size, gamma, batch_size, update_freq, lr, per_cfg, device='cuda'):
        self.name = name
        self.device = device
        self.n_actions = n_actions
        self.ego_dim = ego_dim
        self.nbr_dim = nbr_dim
        self.n_nbr = n_nbr

        self.mean_size = mean_size

        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = update_freq  # target network update
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.step = 0

        self.q_net = DuelingDQN(ego_dim=ego_dim, nbr_dim=nbr_dim, n_actions=n_actions).to(device)
        self.target_net = DuelingDQN(ego_dim=ego_dim, nbr_dim=nbr_dim, n_actions=n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())    

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=lr)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=per_cfg.capacity,
            state_dim=ego_dim + n_nbr*nbr_dim,
            device=device,
            alpha=per_cfg.alpha,
            beta=per_cfg.beta,
            beta_increment=per_cfg.beta_increment
        )

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=True)
            q_values = self.q_net(state[:, :self.ego_dim], state[:, self.ego_dim:].reshape(-1, self.n_nbr, self.nbr_dim))
            return q_values.argmax().item()

    def store(self, s, a, r, s_, done):
        self.replay_buffer.add(s, a, r, s_, done)
        

    def train(self, get_result=False):
        if len(self.replay_buffer) < self.mean_size:
            return
        
        sample_result = self.replay_buffer.sample(self.batch_size)

        if sample_result is None:
            return
        
        states, actions, rewards, next_states, dones, indices, is_weights = sample_result
                # GPU 내에서 직접 reshape (메모리 복사 없음)
        ego_tensor = states[:, :self.ego_dim]
        nbrs_tensor = states[:, self.ego_dim:].reshape(-1, self.n_nbr, self.nbr_dim)
        
        # Q(s,a) 계산
        q_values_tensor = self.q_net(ego_tensor, nbrs_tensor)
        q_a = q_values_tensor.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q target 계산 (Double DQN)
        with torch.no_grad():
            ego_next_tensor = next_states[:, :self.ego_dim]
            nbrs_next_tensor = next_states[:, self.ego_dim:].reshape(-1, self.n_nbr, self.nbr_dim)
            next_q_values_tensor = self.q_net(ego_next_tensor, nbrs_next_tensor)
            next_actions = next_q_values_tensor.argmax(1)
            
            target_q_values_tensor = self.target_net(ego_next_tensor, nbrs_next_tensor)
            q_target = target_q_values_tensor.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target = rewards + self.gamma * q_target * (~dones)
        
        # TD-error 계산
        td_errors = q_a - target
        
        # IS weights 적용한 loss (이미 GPU에 있음)
        weighted_loss = (td_errors ** 2) * is_weights
        loss = weighted_loss.mean()
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 우선순위 업데이트 (GPU에서 직접)
        self.replay_buffer.update_priorities(indices, td_errors.detach())
        
        # β 값 업데이트
        self.replay_buffer.update_beta()
    
                
        self.step += 1

        if self.step % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        if get_result:
            per_state_std, per_state_mean= torch.std_mean(q_values_tensor, dim=1)

            avg_std = np.mean(per_state_std.detach().cpu().numpy())
            avg_mean = np.mean(per_state_mean.detach().cpu().numpy())
            avg_td_error = np.mean(td_errors.detach().cpu().numpy())

            data = {
                "step": self.step, 
                "Q-std": avg_std,
                "Q-mean": avg_mean,
                "TD-error": avg_td_error,
                "beta": self.replay_buffer.beta,
            }
            return data
            # print(f"name: {self.name}, step: {self.step}, Q-variance = {avg_variance:.4f}, "
            #     f"Q-mean = {avg_mean:.4f}, TD-error = {avg_td_error:.4f}, "
            #     f"β = {self.replay_buffer.beta:.3f}")
            # print("-" * 70)
        return None