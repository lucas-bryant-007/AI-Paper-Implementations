import torch
from torch import nn
from collections import deque


class QNetwork(nn.module):
    def __init__(self, input_dims, n_actions):
        super(QNetwork, self).__init__()