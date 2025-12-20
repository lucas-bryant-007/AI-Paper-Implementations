import time
import torch

from agent import DQNAgent
from config import DQNConfig
from env_wrappers import make_env
from network import QNetwork
from replay import ReplayMemory


def main():
    """
    Simple evaluation loop.
    NOTE: This assumes you add model saving/loading if you want persistence.
    For quick checks, you can just run after training in the same session,
    or extend train.py to torch.save(agent.q.state_dict(), "model.pt").
    """
    cfg = DQNConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(
        render_mode="human",
        frame_stack=cfg.frame_stack,
        obs_size=cfg.obs_size,
        continuous=False,
    )

    q_net = QNetwork(n_actions=env.action_space.n, input_channels=cfg.frame_stack).to(device)

    # Placeholder: load weights if you saved them
    # q_net.load_state_dict(torch.load("model.pt", map_location=device))

    # Replay is not used in evaluation, but agent API expects it
    agent = DQNAgent(q_net, env.action_space.n, ReplayMemory(1), cfg, device)

    episodes = 5
    total_steps = 0

    for ep in range(episodes):
        state, info = env.reset()
        ep_reward = 0.0

        done = False
        while not done:
            total_steps += 1
            # Pure greedy action (epsilon=0) for evaluation
            action = agent.act(state, step=10**12)  # effectively eps_end
            state, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            done = term or trunc
            time.sleep(0.01)

        print(f"[EVAL] ep={ep} reward={ep_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
