# buffer.py
# storage class to hold trajectories collected during episode

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class RolloutBuffer:
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    last_value: float = 0.0

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state); self.actions.append(action)
        self.log_probs.append(log_prob); self.rewards.append(reward)
        self.values.append(value); self.dones.append(done)

    def clear(self):
        for lst in (self.states, self.actions, self.log_probs,
                    self.rewards, self.values, self.dones):
            lst.clear()
        self.last_value = 0.0

    def __len__(self):
        return len(self.rewards)