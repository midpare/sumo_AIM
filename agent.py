import random
import numpy as np
from collections import deque
from duelingDQN import DuelingDQN
import torch
import torch.nn.functional as F
from enum import Enum, auto


class AgentType(Enum):
    LEFT = "left"
    STRAIGHT = "straight" 
    RIGHT = "right"

class Action(Enum):
    STOP = 0
    GO = 1 


class ScenarioMemory:
    def __init__(self, capacity=1000, temperature=1.0):
        self.capacity = capacity
        self.temperature = temperature
        self.memory = []
        self.scores = []

    def add(self, episode_transitions, score):
        if len(self.memory) >= self.capacity:
            print("full!")
            self.memory.pop(0)
            self.scores.pop(0)
        self.memory.append(episode_transitions)
        self.scores.append(score)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return []

        priorities = np.exp(-np.array(self.scores) / self.temperature)
        probs = priorities / priorities.sum()
        sampled_episodes = random.choices(self.memory, weights=probs, k=batch_size)
        transitions = [random.choice(ep) for ep in sampled_episodes]
        return transitions

    def __len__(self):
        return len(self.memory)


# class D3QNAgent:
#     def __init__(self, name, input_shape, n_actions, device='cuda'):
#         self.name = name
#         self.device = device
#         self.n_actions = n_actions

#         self.q_net = DuelingDQN(input_dim=input_shape, n_actions=n_actions).to(device)
#         self.target_net = DuelingDQN(input_dim=input_shape, n_actions=n_actions).to(device)

#         self.target_net.load_state_dict(self.q_net.state_dict())

#         self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=1e-4)

#         self.scenario_memory = ScenarioMemory(capacity=10000, temperature=1.0)

#         self.gamma = 0.99
#         self.batch_size = 256
#         self.update_freq = 1000  # target network update

#         self.step = 0

#     def select_action(self, state, epsilon):
#         if random.random() < epsilon:
#             return random.randint(0, self.n_actions - 1)
#         with torch.no_grad():
#             state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             q_values = self.q_net(state)
#             return q_values.argmax().item()

#     def store(self, transition, score):
#         self.scenario_memory.add(transition, score)

#     def train(self):
#         if len(self.scenario_memory) < self.batch_size:
#             return

#         batch = self.scenario_memory.sample(self.batch_size)
#         print(batch)
#         s, a, r, s_, d = zip(*batch)

#         s = torch.FloatTensor(np.array(s)).to(self.device)
#         a = torch.LongTensor(a).to(self.device)
#         r = torch.FloatTensor(r).to(self.device)
#         s_ = torch.FloatTensor(np.array(s_)).to(self.device)
#         d = torch.FloatTensor(d).to(self.device)

#         # Q(s,a)
#         q_values = self.q_net(s)
#         q_a = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

#         # Q target
#         with torch.no_grad():
#             next_actions = self.q_net(s_).argmax(1)
#             q_target = self.target_net(s_).gather(1, next_actions.unsqueeze(1)).squeeze(1)
#             target = r + self.gamma * q_target * (1 - d)

#         loss = F.mse_loss(q_a, target)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.step += 1
#         if self.step % self.update_freq == 0:
#             self.target_net.load_state_dict(self.q_net.state_dict())

#         return loss.item()

class D3QNAgent:
    def __init__(self, name, input_shape, n_actions, device='cuda'):
        self.name = name
        self.device = device
        self.n_actions = n_actions

        self.q_net = DuelingDQN(input_dim=input_shape, n_actions=n_actions).to(device)
        self.target_net = DuelingDQN(input_dim=input_shape, n_actions=n_actions).to(device)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=1e-4)
        self.replay_buffer = deque(maxlen=int(1.5e5))

        self.gamma = 0.99
        self.batch_size = 256
        self.update_freq = 1000  # target network update

        self.step = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            return q_values.argmax().item()

    def store(self, s, a, r, s_, done):
        self.replay_buffer.append((s, a, r, s_, done))

    def train(self, print_q):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a, r, s_, d = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_ = torch.FloatTensor(np.array(s_)).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Q(s,a)
        q_values = self.q_net(s)
        q_a = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

        # Q target
        with torch.no_grad():
            next_actions = self.q_net(s_).argmax(1)
            q_target = self.target_net(s_).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * q_target * (1 - d)

        if not print_q:
            print(q_a[:10])
            print(r[:10])
            print(target[:10])
            print("-"*50)
        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
            