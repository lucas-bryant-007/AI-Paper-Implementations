from dataclasses import dataclass


@dataclass(frozen=True)
class DQNConfig:
    # RL
    gamma: float = 0.99

    # Optimization
    lr: float = 1e-3
    batch_size: int = 32
    max_grad_norm: float = 10.0

    # Replay
    buffer_size: int = 100_000
    learning_starts: int = 10_000  # do not train until buffer has this many transitions

    # Update cadence
    train_freq: int = 4  # train every N environment steps
    target_update_interval: int = 2_000  # hard update target net every N env steps

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 100_000  # linear anneal

    # Episodes / steps
    episodes: int = 500
    max_steps_per_episode: int = 2_000

    # Env / obs
    frame_stack: int = 4
    obs_size: int = 84  # 84x84
    reward_clip: bool = False  # CarRacing reward is not Atari-like; default False
