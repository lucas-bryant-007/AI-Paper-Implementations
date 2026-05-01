# config.py
# puts all hyperparams in one place

from dataclasses import dataclass
from pathlib import Path

@dataclass
class PPOConfig:
    # env
    env_id: str = "BipedalWalker-v3"

    # training
    num_iterations: int = 500
    rollout_length: int = 8192
    k_epochs: int = 5

    # ppo hyperparams
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5 # weight on value loss
    entropy_coef: float = 0.01 # weight on entropy bonus
    max_grad_norm: float = 0.5 # gradient clipping 

    # optimization
    lr: float = 3e-4

    # i/o
    save_dir: Path = Path("model_files")
    log_path: Path = Path("logs/training.csv")

    # evaluation
    num_eval_games = 20