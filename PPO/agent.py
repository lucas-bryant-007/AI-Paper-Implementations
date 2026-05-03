# agent.py
# contains update logic, loss function, action selection

from pathlib import Path
from typing import List, Tuple
 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
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

        self.is_discrete = isinstance(env.action_space, Discrete)
        self.is_continuous = isinstance(env.action_space, Box)

        if self.is_continuous:
            self.action_low = torch.as_tensor(
                env.action_space.low, dtype=torch.float32, device=device
            )
            self.action_high = torch.as_tensor(
                env.action_space.high, dtype=torch.float32, device=device
            )
 
    # --- Rollout collection --------------------------------------------------
 
    @torch.no_grad()
    def _step_policy(self, state_t: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-prob and value (no grad)."""
        dist = self.actor(state_t) # gets distribution from our forward of our actor
        raw_action = dist.sample() # sample action from our distribution
        log_prob = dist.log_prob(raw_action) # gets prob of our raw action
        value = self.critic(state_t) # gets critic value

        if self.is_discrete:
            env_action = int(raw_action.item())
            stored_action = raw_action.squeeze(0).cpu()
        
        elif self.is_continuous:
            stored_action = raw_action.squeeze(0).cpu()
            clipped_action = torch.clamp(
                raw_action.squeeze(0),
                self.action_low,
                self.action_high
            )
            env_action = clipped_action.cpu().numpy()
        
        return env_action, stored_action, log_prob, value
        
 
    def collect_rollout(self, buffer: RolloutBuffer) -> List[float]:
        """Roll out `cfg.rollout_length` steps. Returns finished-episode returns."""
        buffer.clear()
        episode_returns: List[float] = []
        ep_return = 0.0
 
        state, _ = self.env.reset()
        for _ in range(self.cfg.rollout_length):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            env_action, stored_action, log_prob, value = self._step_policy(state_t.unsqueeze(0))
 
            next_state, reward, terminated, truncated, _ = self.env.step(env_action)
            done = terminated or truncated
            
            with torch.no_grad():
                next_state_val = self.critic(torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0))


            buffer.add(
                state=state_t.cpu(),
                action=stored_action,
                log_prob=log_prob.cpu().squeeze(),
                reward=float(reward),
                value=value.cpu().squeeze(),
                next_value=next_state_val.cpu().squeeze(),
                done=done,
                terminated=terminated,
            )

            ep_return += reward
            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0
                state, _ = self.env.reset()
            else:
                state = next_state
 
        return episode_returns
 
    # --- Advantage estimation ------------------------------------------------
 
    def _compute_gae(self, buffer: RolloutBuffer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation. Returns (advantages, returns)."""
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
        values = torch.stack(buffer.values).float()
        dones = torch.tensor(buffer.dones, dtype=torch.float32)
        terminates = torch.tensor(buffer.terminates, dtype=torch.float32)

        masks = 1.0 - dones  # 0 wherever an episode ended at step t
        masks_t = 1.0 - terminates
 
        next_values = torch.stack(buffer.next_values).float()
        deltas = rewards + self.cfg.gamma * next_values * masks_t - values
 
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

        if self.is_discrete:
            actions = torch.stack(buffer.actions).long().to(self.device)
        elif self.is_continuous:
            actions = torch.stack(buffer.actions).float().to(self.device)

        old_log_probs = torch.stack(buffer.log_probs).to(self.device)
 
        params = list(self.actor.parameters()) + list(self.critic.parameters())

        batch_size = len(buffer)
        minibatch_size = self.cfg.minibatch_size

        for _ in range(self.cfg.k_epochs):
            indices = torch.randperm(batch_size, device=self.device) # shuffle our indices for minibatch sampling, so better learning signal than local extrema that consecutive states can sometimes provide

            for start in range(0, batch_size, minibatch_size):
                end = start+minibatch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx] # get log probs at minibatch indices, not recomputed because we already computed log_probs
                mb_advantages = advantages[mb_idx] # fine to shuffle advantage, update signal only relies on advantages, not consecutiveness
                mb_returns = returns[mb_idx]

                dist = self.actor(mb_states)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs) # prob of action under new policy divided by prob under old policy

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(self.critic(mb_states), mb_returns)

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
        
        action = self.actor(state_t).sample()

        if self.is_discrete:
            return int(action.item())
        elif self.is_continuous:
            action = action.squeeze(0)
            action = torch.clamp(action, self.action_low, self.action_high)
            return action.cpu().numpy()
        
 
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