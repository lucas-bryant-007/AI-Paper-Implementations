# main.py
# entry point, initialize gym, run train loop

import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import agent
import buffer
import model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_ppo(run_name, lr, seed=None):
    print(f"running PPO on CartPole | run={run_name} | lr={lr} | seed={seed}")

    set_seed(seed)

    env = gym.make("CartPole-v1")

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    T = 2048
    num_iterations = 100
    gamma = 0.99
    k_epochs = 10
    eps = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    buffer_ = buffer.rollout_buffer()

    actor = model.actor(env).to(device)
    critic = model.critic(env).to(device)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr
    )

    ppo_agent = agent.ppo_agent(
        env,
        actor,
        critic,
        optimizer,
        gamma=gamma,
        k_epochs=k_epochs,
        epsilon=eps,
        device=device
    )

    rows = []

    for iteration in range(num_iterations):
        episode_rewards = ppo_agent.run_for_T_timesteps(buffer_, T)
        ppo_agent.update(buffer_)

        if len(episode_rewards) > 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
        else:
            avg_reward = float("nan")

        row = {
            "run_name": run_name,
            "lr": lr,
            "seed": seed,
            "iteration": iteration,
            "avg_reward": avg_reward,
            "num_episodes": len(episode_rewards),
            "T": T,
            "gamma": gamma,
            "k_epochs": k_epochs,
            "epsilon": eps,
        }

        rows.append(row)

        if iteration % 10 == 0:
            print(
                f"Run {run_name} | Seed {seed} | Iteration {iteration} "
                f"| Avg reward: {avg_reward:.2f} | Episodes: {len(episode_rewards)}"
            )

    env.close()
    return rows


def save_results_csv(rows, path, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "lr",
        "seed",
        "iteration",
        "avg_reward",
        "num_episodes",
        "T",
        "gamma",
        "k_epochs",
        "epsilon",
    ]

    file_exists = path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"

    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not append or not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print(f"Saved results to {path}")

if __name__ == "__main__":
    seeds_per_run = 10
    start_at_seed = 10

    all_rows = []

    for seed in range(start_at_seed, start_at_seed + seeds_per_run):
        all_rows.extend(run_ppo(run_name="vanilla", lr=3e-4, seed=seed))
        all_rows.extend(run_ppo(run_name="increased_lr", lr=6e-4, seed=seed))

    save_results_csv(
        all_rows,
        "results/ppo_cartpole_results.csv",
        append=True
    )