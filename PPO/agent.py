# agent.py
# contains update logic, loss function, action selection

import numpy as np
import torch

import buffer


class ppo_agent():
    def __init__(self, env, actor, critic, optim, gamma, k_epochs, epsilon, gae_param, device):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.epsilon = epsilon
        self.gae_param = gae_param
        self.device = device

        self.actor.to(self.device)
        self.critic.to(self.device)


    def select_action(self, state, buffer: buffer.rollout_buffer):
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor.forward(state)
            value = self.critic.forward(state)

        action_item = action.item()
        log_prob_det = log_prob.detach()

        next_state, reward, terminated, trunc, _ = self.env.step(action_item)
        done = terminated or trunc

        next_state_tensor = torch.tensor(next_state, dtype=torch.float).to(self.device).unsqueeze(0)
        with torch.no_grad():
            next_state_value = self.critic.forward(next_state_tensor)

        buffer.states.append(state.detach().cpu())
        buffer.actions.append(action_item)
        buffer.log_probs.append(log_prob_det.cpu())
        buffer.rewards.append(reward)
        buffer.state_values.append(value.detach().cpu().squeeze())
        buffer.next_state_values.append(next_state_value.detach().cpu().squeeze())        
        buffer.dones.append(done)

        return next_state, done

    def compute_returns(self, buffer: buffer.rollout_buffer):
        returns = []
        discounted_reward = 0

        for reward, done in zip(reversed(buffer.rewards), reversed(buffer.dones)):
            if done:
                discounted_reward = 0
            
            discounted_reward *= self.gamma
            discounted_reward += reward

            returns.append(discounted_reward)
        
        return returns[::-1]
    
    def compute_advantage(self, buffer: buffer.rollout_buffer):
        returns = torch.tensor(self.compute_returns(buffer)).to(self.device)
        values = torch.stack(buffer.state_values).to(self.device)
        
        advantages = torch.subtract(returns, values)
        
        advantages_std, advantages_mean = torch.std_mean(advantages)

        advantages_norm = (advantages - advantages_mean)/(advantages_std + 1e-8)

        return advantages_norm

    def compute_advantage_gae(self, buffer: buffer.rollout_buffer):
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32).to(self.device)
        values = torch.stack(buffer.state_values).to(self.device)
        values_next = torch.stack(buffer.next_state_values).to(self.device)
        not_done = 1.0 - torch.tensor(buffer.dones, dtype=torch.float32).to(self.device)
        deltas = rewards + self.gamma*values_next*not_done - values

        advantages = []
        discounted_return = 0

        for delta, done in zip(reversed(deltas), reversed(buffer.dones)):
            if done:
                discounted_return = 0
            
            
            discounted_return = delta + self.gamma * self.gae_param * discounted_return
            advantages.append(discounted_return)
        
        raw_gae_advantages = torch.stack(advantages[::-1])

                
        advantages_std, advantages_mean = torch.std_mean(raw_gae_advantages)

        gae_advantages_norm = (raw_gae_advantages - advantages_mean)/(advantages_std + 1e-8)

        return raw_gae_advantages, gae_advantages_norm

    def update(self, buffer: buffer.rollout_buffer):

        # 1. Precalculate targets and advantages once
        raw_advantages, advantages = self.compute_advantage_gae(buffer)
        advantages = advantages.detach()

        old_states = torch.cat(buffer.states).to(self.device)
        old_actions = torch.tensor(buffer.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.cat(buffer.log_probs).detach().to(self.device).squeeze(-1)
        
        values = torch.stack(buffer.state_values).to(self.device).detach()
        returns = (raw_advantages + values).detach()
        
        for epoch in range(self.k_epochs):
            
            log_probs, dist_entropy = self.actor.get_probs(old_states, old_actions)
            log_probs = log_probs.squeeze(-1)
            state_values = self.critic(old_states).squeeze(-1)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(state_values, returns)
            entropy_loss = dist_entropy.mean()
            
            self.optim.zero_grad()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            loss.backward()
            self.optim.step()

        buffer.clear()

    def run_for_T_timesteps(self, buffer, T):
        state, info = self.env.reset()

        episode_reward = 0
        completed_episode_rewards = []

        for timestep in range(T):
            state, done = self.select_action(state, buffer)
            
            episode_reward += buffer.rewards[-1]
            
            if done:
                completed_episode_rewards.append(episode_reward)
                episode_reward = 0

                state, info = self.env.reset()

        return completed_episode_rewards
        
