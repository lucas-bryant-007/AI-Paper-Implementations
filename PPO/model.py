# model.py
# defines actor (policy) and critic (value function) neural networks using pytorch

import torch
import torch.nn as nn
import gymnasium as gym

class actor(nn.Module):
    def __init__(self, env: gym.Env):
        super(actor, self).__init__()

        self.linear1 = nn.Linear(env.observation_space.shape[0], 64)
        self.activation1 = nn.Tanh()
        self.linear2 = nn.Linear(64, 64)
        self.activation2 = nn.Tanh()
        self.linear3 = nn.Linear(64, env.action_space.n)
    
    def forward(self, X):
        X = self.linear1(X)
        X = self.activation1(X)
        X = self.linear2(X)
        X = self.activation2(X)
        X = self.linear3(X)
        X = torch.distributions.Categorical(logits=X)
        action = X.sample()
        return action, X.log_prob(action)
    
    def get_probs(self, state, action):
        X = self.linear1(state)
        X = self.activation1(X)
        X = self.linear2(X)
        X = self.activation2(X)
        logits = self.linear3(X)
        dist = torch.distributions.Categorical(logits=logits)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_probs, dist_entropy

class critic(nn.Module):
    def __init__(self, env: gym.Env):
        super(critic, self).__init__()

        self.linear1 = nn.Linear(env.observation_space.shape[0], 64)
        self.activation1 = nn.Tanh()
        self.linear2 = nn.Linear(64, 64)
        self.activation2 = nn.Tanh()
        self.linear3 = nn.Linear(64, 1)
    
    def forward(self, X):
        X = self.linear1(X)
        X = self.activation1(X)
        X = self.linear2(X)
        X = self.activation2(X)
        X = self.linear3(X)
        return X


