# agent.py
# contains update logic, loss function, action selection

from pathlib import Path
from typing import List, Tuple
 
import gymnasium as gym
import torch
import torch.nn.functional as F
 
from buffer import RolloutBuffer
from config import PPOConfig
from model import Actor, Critic
 
 
class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        actor: Actor,
        critic: Critic,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
        device: torch.device,
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.cfg = config
        self.device = device
 
    # --- Rollout collection --------------------------------------------------
 
    @torch.no_grad()
    def _step_policy(self, state_t: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-prob and value (no grad)."""
        dist = self.actor(state_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action), self.critic(state_t)
 
    def collect_rollout(self, buffer: RolloutBuffer) -> List[float]:
        """Roll out `cfg.rollout_length` steps. Returns finished-episode returns."""
        buffer.clear()
        episode_returns: List[float] = []
        ep_return = 0.0
 
        state, _ = self.env.reset()
        for _ in range(self.cfg.rollout_length):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, log_prob, value = self._step_policy(state_t.unsqueeze(0))
 
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
 
            buffer.add(
                state=state_t.cpu(),
                action=action,
                log_prob=log_prob.cpu().squeeze(),
                reward=float(reward),
                value=value.cpu().squeeze(),
                done=done,
            )
 
            ep_return += reward
            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0
                state, _ = self.env.reset()
            else:
                state = next_state
 
        # Bootstrap value at the end of the rollout (only this once per rollout).
        with torch.no_grad():
            last_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            buffer.last_value = float(self.critic(last_t.unsqueeze(0)).item())
        return episode_returns
 
    # --- Advantage estimation ------------------------------------------------
 
    def _compute_gae(self, buffer: RolloutBuffer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation. Returns (advantages, returns)."""
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
        values = torch.stack(buffer.values).float()
        dones = torch.tensor(buffer.dones, dtype=torch.float32)
        masks = 1.0 - dones  # 0 wherever an episode ended at step t
 
        # next_values[t] = V(s_{t+1}); the last next-value is the bootstrap.
        next_values = torch.cat([values[1:], torch.tensor([buffer.last_value])])
        deltas = rewards + self.cfg.gamma * next_values * masks - values
 
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(buffer))):
            gae = deltas[t] + self.cfg.gamma * self.cfg.gae_lambda * masks[t] * gae
            advantages[t] = gae
 
        returns = advantages + values
        return advantages.to(self.device), returns.to(self.device)
 
    # --- Update --------------------------------------------------------------
 
    def update(self, buffer: RolloutBuffer) -> None:
        advantages, returns = self._compute_gae(buffer)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
        states = torch.stack(buffer.states).to(self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(buffer.log_probs).to(self.device)
 
        params = list(self.actor.parameters()) + list(self.critic.parameters())
 
        for _ in range(self.cfg.k_epochs):
            dist = self.actor(states)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
 
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(self.critic(states), returns)
 
            loss = (
                policy_loss
                + self.cfg.value_coef * value_loss
                - self.cfg.entropy_coef * entropy
            )
 
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
            self.optimizer.step()
 
    # --- Inference -----------------------------------------------------------
 
    @torch.no_grad()
    def act(self, state) -> int:
        """Sample one action from the current policy (used at eval time)."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.actor(state_t).sample().item())
 
    # --- Persistence ---------------------------------------------------------
 
    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            save_dir / "ppo_checkpoint.pt",
        )
 
    def load(self, save_dir: Path) -> None:
        ckpt = torch.load(Path(save_dir) / "ppo_checkpoint.pt", map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])