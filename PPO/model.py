# model.py
# defines actor (policy) and critic (value function) neural networks using pytorch

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

def _mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim)
    )

def _orthagonal_init(module, gain):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

class Actor(nn.Module):
    def __init__(self, env, hidden=64):
        super().__init__()
        self.net = _mlp(env.observation_space.shape[0], env.action_space.n, hidden)
        _orthagonal_init(self.net, gain=0.01)
    
    def forward(self, obs):
        return Categorical(logits=self.net(obs))

class Critic(nn.Module):
    def __init__(self, env, hidden=64):
        super().__init__()
        self.net = _mlp(env.observation_space.shape[0], 1, hidden)
        _orthagonal_init(self.net, gain=1.0)
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)
