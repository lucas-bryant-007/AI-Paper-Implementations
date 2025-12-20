from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import DQNConfig
from replay import ReplayMemory
from schedules import LinearEpsilonSchedule


class DQNAgent:
    """
    Standard DQN:
      - Experience replay
      - Target network (hard update)
      - Huber loss
      - Gradient clipping
    """

    def __init__(
        self,
        q_network: torch.nn.Module,
        n_actions: int,
        replay: ReplayMemory,
        cfg: DQNConfig,
        device: torch.device,
    ):
        self.q = q_network.to(device)
        self.target = copy.deepcopy(self.q).to(device)
        self.target.eval()

        self.n_actions = n_actions
        self.replay = replay
        self.cfg = cfg
        self.device = device

        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.eps_sched = LinearEpsilonSchedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)

    @torch.no_grad()
    def act(self, state: np.ndarray, step: int) -> int:
        eps = self.eps_sched(step)
        if np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))

        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,C,H,W)
        q_values = self.q(x)  # (1, A)
        return int(q_values.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, terminated: bool) -> None:
        self.replay.add(state, action, reward, next_state, terminated)

    def update(self, step: int) -> Optional[float]:
        if len(self.replay) < self.cfg.learning_starts:
            return None
        if step % self.cfg.train_freq != 0:
            return None

        states, actions, rewards, next_states, terminated = self.replay.sample(
            self.cfg.batch_size, device=self.device
        )

        # Q(s,a)
        q_sa = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # DQN target using target network
            next_q = self.target(next_states).max(dim=1).values
            target = rewards + self.cfg.gamma * next_q * (1.0 - terminated)

        loss = F.smooth_l1_loss(q_sa, target)  # Huber

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optim.step()

        if step % self.cfg.target_update_interval == 0:
            self.target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def epsilon(self, step: int) -> float:
        return float(self.eps_sched(step))
