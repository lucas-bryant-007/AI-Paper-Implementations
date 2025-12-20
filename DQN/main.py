import torch

import dqn

N = 10_000


net = dqn.QNetwork(n_actions=6, input_channels=4)
x = torch.randn(32, 4, 84, 84)
q = net(x)
print(q.shape)