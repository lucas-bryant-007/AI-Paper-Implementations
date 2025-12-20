from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    state: np.ndarray       # (C,H,W) float32
    action: int
    reward: float
    next_state: np.ndarray  # (C,H,W) float32
    terminated: bool        # true terminal only (NOT time-limit truncation)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                state=state.astype(np.float32, copy=False),
                action=int(action),
                reward=float(reward),
                next_state=next_state.astype(np.float32, copy=False),
                terminated=bool(terminated),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.int64, device=device)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=device)
        terminated = torch.as_tensor([b.terminated for b in batch], dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, terminated
