# model.py
# defines actor (policy) and critic (value function) neural networks using pytorch

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent

def _mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim)
    )

def _orthagonal_init(module, hidden_gain, output_gain):
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)] # Makes a list of all linear layers in the network
    
    for layer in linear_layers[:-1]: # for all but last layer in network
        nn.init.orthogonal_(layer.weight, gain=hidden_gain)
        nn.init.zeros_(layer.bias)
    
    output_layer = linear_layers[-1] # get LAST layer of network
    nn.init.orthogonal_(output_layer.weight, gain=output_gain) # give only last layer the output gain
    nn.init.zeros_(output_layer.bias)

class Actor(nn.Module):
    def __init__(self, env: gym.Env, hidden=64):
        super().__init__()
        
        obs_shape = env.observation_space.shape # get entire obs shape as tuple
        obs_dim = int(np.prod(obs_shape)) # multiply tuple out to get total flattened obs size

        self.is_discrete = isinstance(env.action_space, Discrete) # tracks if our action space is discrete in gym
        self.is_continuous = isinstance(env.action_space, Box) # bool tracking if continuous action space

        if self.is_discrete: # if discrete action space, set up mlp
            # Do Discrete
            action_n = env.action_space.n
            self.net = _mlp(obs_dim, action_n, hidden)
            _orthagonal_init(self.net, hidden_gain=math.sqrt(2), output_gain=0.01)
        elif self.is_continuous:
            # Do Continuous
            action_dim = env.action_space.shape[0]
            self.net = _mlp(obs_dim, action_dim, hidden)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
            _orthagonal_init(self.net, hidden_gain=math.sqrt(2), output_gain=0.01)

        else:
            raise NotImplementedError(
                f"Unsupported action space: {type(env.action_space)}"
            )


        
    
    def forward(self, obs): 
        obs = obs.flatten(start_dim=1) # flattens observation
        if self.is_discrete:
            logits = self.net(obs)
            return Categorical(logits=logits)
        elif self.is_continuous:
            mean = self.net(obs)
            std = self.log_std.exp().expand_as(mean)
            return Independent(Normal(mean, std), 1)
            
class Critic(nn.Module):
    def __init__(self, env, hidden=64):
        super().__init__()
        
        obs_shape = env.observation_space.shape # get entire obs shape as tuple
        obs_dim = int(np.prod(obs_shape)) # multiply tuple out to get total flattened obs size

        self.net = _mlp(obs_dim, 1, hidden)
        _orthagonal_init(self.net, hidden_gain=math.sqrt(2), output_gain=1.00)
    
    def forward(self, obs):
        obs = obs.flatten(start_dim=1) # flattens observation
        return self.net(obs).squeeze(-1)