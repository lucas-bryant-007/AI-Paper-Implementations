
import random
from collections import deque, namedtuple

import torch
from torch import nn


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayMemory():
    def __init__(self, capacity:int):
        self.buffer = deque(maxlen=capacity)
    def len(self):
        return len(self.buffer)
    def add(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
    def sample(self, batch_size: int, device=None):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # To tensors for torch
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

class QNetwork(nn.Module):
    def __init__(self, n_actions: int, input_channels: int = 4):
        super(QNetwork, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.features(x)
        q = self.q_head(x)
        return q
    
