import random, time
import numpy as np
from duelingDQN import DuelingDQN
import torch
from enum import Enum


class AgentType(Enum):
    LEFT = "left"
    STRAIGHT = "straight" 
    RIGHT = "right"

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha              # 우선순위 강도
        self.beta = beta                # IS weights 강도  
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """새로운 경험 추가"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        # 새로운 경험은 최대 우선순위로 설정
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        """우선순위 기반 샘플링"""
        if len(self.buffer) < batch_size:
            return None, None, None
            
        # 우선순위 계산
        priorities = self.priorities[:len(self.buffer)]
        priorities = priorities ** self.alpha
        
        # 확률 분포 생성
        total_priority = priorities.sum()
        probs = priorities / total_priority
        
        # 우선순위 기반 샘플링
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 경험들 추출
        experiences = [self.buffer[i] for i in indices]
        
        # Importance Sampling weights 계산
        weights = self._calculate_is_weights(probs, indices, batch_size)

        return experiences, indices, weights
    
    def _calculate_is_weights(self, probs, indices, batch_size):
        """Importance Sampling weights 계산"""
        N = len(self.buffer)
        weights = []
        
        for i in indices:
            weight = (N * probs[i]) ** (-self.beta)
            weights.append(weight)
        
        # 정규화 (최대값으로 나눠서 안정성 확보)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.max()
        
        return weights
    
    def update_priorities(self, indices, td_errors):
        """학습 후 우선순위 업데이트"""
        for i, td_error in zip(indices, td_errors):
            # 최소 우선순위 설정 (0이 되지 않도록)
            priority = abs(td_error) + 1e-6
            self.priorities[i] = priority
            
        # 최대 우선순위 업데이트
        self.max_priority = max(self.max_priority, np.max(np.abs(td_errors)) + 1e-6)
    
    def update_beta(self):
        """β 값 점진적 증가"""
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
    
    def __len__(self):
        return len(self.buffer)

class D3QNAgent:
    def __init__(self, name, ego_dim, nbr_dim, n_nbr, n_actions, mean_size, gamma, batch_size, update_freq, lr, per_cfg, device='cuda'):
        self.name = name
        self.device = device
        self.n_actions = n_actions

        self.q_net = DuelingDQN(ego_dim=ego_dim, nbr_dim=nbr_dim, n_actions=n_actions).to(device)
        self.target_net = DuelingDQN(ego_dim=ego_dim, nbr_dim=nbr_dim, n_actions=n_actions).to(device)
        self.ego_dim = ego_dim
        self.nbr_dim = nbr_dim
        self.n_nbr = n_nbr

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=1e-4)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=int(per_cfg.capacity),
            alpha=per_cfg.alpha,
            beta=per_cfg.beta,
            beta_increment=per_cfg.beta_increment
        )

        self.mean_size = mean_size

        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = update_freq  # target network update
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        self.step = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state[:, :self.ego_dim], state[:, self.ego_dim:].reshape(-1, self.n_nbr, self.nbr_dim))
            return q_values.argmax().item()

    def store(self, s, a, r, s_, done):
        self.replay_buffer.add(s, a, r, s_, done)

    def train(self, store_result=False):
        if len(self.replay_buffer) < self.mean_size:
            return

        sample_result = self.replay_buffer.sample(self.batch_size)
        if sample_result[0] is None:
            return None
            
        experiences, indices, is_weights = sample_result
        
        s, a, r, s_, d = zip(*experiences)

        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_ = torch.FloatTensor(np.array(s_)).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        is_weights_tensor = torch.FloatTensor(is_weights).to(self.device)

        # print(f"np to tensor! in {time.time() - train_start:.5f}")

        ego     = s[:, :self.ego_dim]          # ego state: (B, )
        nbrs    = s[:, self.ego_dim:].reshape(-1, self.n_nbr, self.nbr_dim)   # neighbor states: (B, 3, 2)

        # Q(s,a)
        q_values = self.q_net(ego, nbrs)
        q_a = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

        # Q target
        # print(f"get Q! in {time.time() - train_start:.5f}")

        with torch.no_grad():
            ego_    = s_[:, :self.ego_dim]         # next state ego
            nbrs_   = s_[:, self.ego_dim:].reshape(-1, self.n_nbr, self.nbr_dim)  # next state neighbors
            next_actions = self.q_net(ego_, nbrs_).argmax(1)
            q_target = self.target_net(ego_, nbrs_).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * q_target * (1 - d)
        
        td_errors = q_a - target

        # IS weights 적용한 loss 계산
        weighted_loss = (td_errors ** 2) * is_weights_tensor
        loss = weighted_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 우선순위 업데이트
        td_errors_np = torch.abs(td_errors).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors_np)
        
        # β 값 업데이트
        self.step += 1
        self.replay_buffer.update_beta()

        if self.step % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        if store_result:
            per_state_std, per_state_mean= torch.std_mean(q_values, dim=1)

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