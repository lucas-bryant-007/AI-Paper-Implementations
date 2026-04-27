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


### WARNING: PPO CURRENTLY ONLY WORKS FOR DISCRETE ACTION SPACE
# TODO: add support for continuous action space, multiple input sizes/shapes, multiple output action shapes

def run_ppo(run_name, lr):
    print(f"running PPO on Acrobot | run={run_name} | lr={lr}")

    env = gym.make('Acrobot-v1')

    env.reset()


    T = 2048
    num_iterations = 100
    gamma = 0.99
    k_epochs = 10
    eps = 0.2
    gae_param = 0.95

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
        gae_param=gae_param,
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
                f"Run {run_name} | Iteration {iteration} "
                f"| Avg reward: {avg_reward:.2f} | Episodes: {len(episode_rewards)}"
            )

    env.close()
    return rows

if __name__ == "__main__":

        run_ppo(run_name="vanilla_ppo", lr=3e-4)