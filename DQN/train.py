import numpy as np
import torch

from agent import DQNAgent
from config import DQNConfig
from env_wrappers import make_env
from network import QNetwork
from replay import ReplayMemory


def maybe_clip_reward(r: float, enabled: bool) -> float:
    if not enabled:
        return r
    return float(np.clip(r, -1.0, 1.0))


def main():
    cfg = DQNConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train without rendering for throughput
    env = make_env(
        render_mode=None,
        frame_stack=cfg.frame_stack,
        obs_size=cfg.obs_size,
        continuous=False,
    )

    q_net = QNetwork(n_actions=env.action_space.n, input_channels=cfg.frame_stack)
    replay = ReplayMemory(cfg.buffer_size)
    agent = DQNAgent(q_net, env.action_space.n, replay, cfg, device)

    total_steps = 0

    for ep in range(cfg.episodes):
        state, info = env.reset()
        ep_reward = 0.0
        last_loss = None

        for t in range(cfg.max_steps_per_episode):
            total_steps += 1

            action = agent.act(state, total_steps)
            next_state, reward, term, trunc, info = env.step(action)

            reward = maybe_clip_reward(reward, cfg.reward_clip)

            # Important: bootstrap mask should use true termination only.
            agent.observe(state, action, reward, next_state, terminated=term)

            loss = agent.update(total_steps)
            if loss is not None:
                last_loss = loss

            state = next_state
            ep_reward += reward

            if term or trunc:
                break

        print(
            f"ep={ep:4d} steps={total_steps:8d} "
            f"reward={ep_reward:8.1f} eps={agent.epsilon(total_steps):.3f} "
            f"loss={last_loss if last_loss is not None else '—'}"
        )

    env.close()


if __name__ == "__main__":
    main()
