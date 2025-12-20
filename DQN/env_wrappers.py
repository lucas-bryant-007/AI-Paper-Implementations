from __future__ import annotations

from collections import deque
from typing import Deque, Tuple, Any

import cv2
import gymnasium as gym
import numpy as np


class PreprocessFrame(gym.ObservationWrapper):
    """
    CarRacing returns RGB (H,W,3) uint8.
    Convert -> grayscale (84,84) float32 in [0,1].
    """

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(size, size),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32) / 255.0
        obs = obs.mean(axis=2)  # grayscale
        obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return obs.astype(np.float32)


class FrameStack(gym.Wrapper):
    """Stack last k frames -> (k, H, W) float32."""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames: Deque[np.ndarray] = deque(maxlen=k)

        assert isinstance(env.observation_space, gym.spaces.Box)
        h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(k, h, w),
            dtype=np.float32,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, term, trunc, info

    def _get_obs(self) -> np.ndarray:
        return np.stack(self.frames, axis=0).astype(np.float32)


def make_env(
    render_mode: str | None,
    frame_stack: int = 4,
    obs_size: int = 84,
    lap_complete_percent: float = 0.95,
    domain_randomize: bool = False,
    continuous: bool = False,
) -> gym.Env:
    """
    CarRacing-v3 supports discrete actions when continuous=False.
    """
    env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=lap_complete_percent,
        domain_randomize=domain_randomize,
        continuous=continuous,
    )
    env = PreprocessFrame(env, size=obs_size)
    env = FrameStack(env, k=frame_stack)
    return env
