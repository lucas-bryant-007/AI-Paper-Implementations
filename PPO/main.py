# main.py
# entry point, initialize gym, run train loop


import argparse
import csv
from typing import Dict, List

import gymnasium as gym
import torch

from agent import PPOAgent
from buffer import RolloutBuffer
from config import PPOConfig
from model import Actor, Critic


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_agent(env: gym.Env, cfg: PPOConfig, device: torch.device) -> PPOAgent:
    """Construct actor, critic, optimizer, and agent. Shared by train and eval."""
    actor = Actor(env, hidden=64).to(device)
    critic = Critic(env, hidden=64).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr
    )
    return PPOAgent(env, actor, critic, optimizer, cfg, device)


def _write_log(log: List[Dict], cfg: PPOConfig) -> None:
    if not log:
        return
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    print(f"[train] wrote training log to {cfg.log_path}")


def train(cfg: PPOConfig, run_name: str = "vanilla_ppo") -> List[Dict]:
    device = _pick_device()
    print(f"[train] env={cfg.env_id}  device={device}  iters={cfg.num_iterations}  lr={cfg.lr}")

    env = gym.make(cfg.env_id)
    agent = _build_agent(env, cfg, device)
    buffer = RolloutBuffer()

    log: List[Dict] = []
    for it in range(cfg.num_iterations):
        episode_returns = agent.collect_rollout(buffer)
        agent.update(buffer)

        avg_return = (
            sum(episode_returns) / len(episode_returns)
            if episode_returns else float("nan")
        )
        log.append({
            "run_name": run_name,
            "iteration": it,
            "avg_return": avg_return,
            "num_episodes": len(episode_returns),
            "lr": cfg.lr,
        })

        if it % 2 == 0 or it == cfg.num_iterations - 1:
            print(
                f"[train] iter {it:3d} | "
                f"avg_return {avg_return:7.2f} | "
                f"episodes {len(episode_returns)}"
            )

    agent.save(cfg.save_dir)
    print(f"[train] saved checkpoint to {cfg.save_dir}")
    _write_log(log, cfg)

    env.close()
    return log


def evaluate(cfg: PPOConfig, render: bool = True) -> None:
    device = _pick_device()
    env = gym.make(cfg.env_id, render_mode="human" if render else None)
    agent = _build_agent(env, cfg, device)
    agent.load(cfg.save_dir)
    print(f"[eval] loaded checkpoint from {cfg.save_dir}")

    for game in range(cfg.num_eval_games):
        state, _ = env.reset()
        ep_return, done = 0.0, False
        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        print(f"[eval] game {game + 1}/{cfg.num_eval_games}  return {ep_return:.2f}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO on a Gymnasium env.")
    parser.add_argument(
        "mode", choices=["train", "eval", "both"], nargs="?", default="both",
        help="What to run (default: both).",
    )
    parser.add_argument("--run-name", default="vanilla_ppo")
    parser.add_argument("--env", default=None, help="Override env_id.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.env is not None:
        cfg.env_id = args.env
    if args.lr is not None:
        cfg.lr = args.lr

    if args.mode in ("train", "both"):
        train(cfg, run_name=args.run_name)
    if args.mode in ("eval", "both"):
        evaluate(cfg)


if __name__ == "__main__":
    main()