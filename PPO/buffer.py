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
    next_values: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    terminates: List[bool] = field(default_factory=list)

    def add(self, state, action, log_prob, reward, value, next_value, done, terminated):
        self.states.append(state); self.actions.append(action)
        self.log_probs.append(log_prob); self.rewards.append(reward)
        self.values.append(value); self.next_values.append(next_value)
        self.dones.append(done); self.terminates.append(terminated)

    def clear(self):
        for lst in (self.states, self.actions, self.log_probs,
                    self.rewards, self.values, self.next_values, self.dones, self.terminates):
            lst.clear()

    def __len__(self):
        return len(self.rewards)