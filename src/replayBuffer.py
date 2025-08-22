import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, device, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha              # 우선순위 강도
        self.beta = beta                # IS weights 강도  
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        self.states = torch.zeros(capacity, state_dim, device=device, dtype=torch.float32)
        self.actions = torch.zeros(capacity, device=device, dtype=torch.long)
        self.rewards = torch.zeros(capacity, device=device, dtype=torch.float32)
        self.next_states = torch.zeros(capacity, state_dim, device=device, dtype=torch.float32)
        self.dones = torch.zeros(capacity, device=device, dtype=torch.bool)
        
        self.device = device
        self.buffer = []
        self.priorities = torch.ones(capacity, device=device, dtype=torch.float32)       
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """새로운 경험 추가"""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state.astype(np.float32))
        else:
            state = torch.tensor(state, dtype=torch.float32)
            
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state.astype(np.float32))
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32)
        
        self.states[self.pos] = state.to(self.device, non_blocking=True)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state.to(self.device, non_blocking=True)
        self.dones[self.pos] = done
            
        # 새로운 경험은 최대 우선순위로 설정
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        
    def sample(self, batch_size):
        """우선순위 기반 샘플링"""
        if self.size < batch_size:
            return None

        with torch.no_grad():
            valid_priorities = self.priorities[:self.size]
            priorities_alpha = torch.pow(valid_priorities, self.alpha)
            
            probs = priorities_alpha / torch.sum(priorities_alpha)            
            indices = torch.multinomial(probs, batch_size, replacement=True)
            
            # IS weights 계산 (GPU에서)
            sampling_probs = probs[indices]
            weights = torch.pow(self.size * sampling_probs, -self.beta)
            weights = weights / torch.max(weights)  # 정규화
            
            # 배치 데이터 추출 (GPU 메모리 내에서)
            batch_states = self.states[indices]
            batch_actions = self.actions[indices]
            batch_rewards = self.rewards[indices]
            batch_next_states = self.next_states[indices]
            batch_dones = self.dones[indices]
            
            return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights
    
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        # 최대 우선순위 업데이트
        with torch.no_grad():
            if isinstance(td_errors, np.ndarray):
                td_errors = torch.from_numpy(td_errors).to(self.device)
            
            # 새로운 우선순위 계산
            new_priorities = torch.abs(td_errors) + 1e-6
            self.priorities[indices] = new_priorities
            
            # 최대 우선순위 업데이트
            self.max_priority = max(self.max_priority, torch.max(new_priorities).item())
    
    
    def update_beta(self):
        """β 값 점진적 증가"""
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
    
    def __len__(self):
        return self.size
