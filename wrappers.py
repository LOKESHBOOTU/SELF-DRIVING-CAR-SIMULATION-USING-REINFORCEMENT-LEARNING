"""
wrappers.py — Custom Gymnasium Wrappers for CarRacing-v2
=========================================================
Wrappers modify the environment's observations, actions, or rewards
WITHOUT touching the core logic. Stack them like layers.

CarRacingWrapper (this file)
  → GrayScaleObservation  (gymnasium built-in)
  → ResizeObservation     (gymnasium built-in)
  → VecFrameStack         (SB3 VecEnv wrapper)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CarRacingWrapper(gym.Wrapper):
    """
    Custom wrapper for CarRacing-v2 that:

    1. Skips the initial zoom-in animation (first ~50 frames are useless)
    2. Clips rewards to [-1, 1] to stabilise training
    3. Terminates an episode early if the agent goes off-track for too long
       (avoids wasting rollout steps on hopeless episodes)
    4. Normalises continuous actions to [-1, 1] (already the case, but made explicit)

    Args:
        env:                  The wrapped gymnasium environment.
        skip_frames:          Frames to skip at episode start (zoom animation).
        max_negative_steps:   Stop episode after this many consecutive negative-reward steps.
        reward_clip:          Clip rewards to this absolute value.
    """

    def __init__(
        self,
        env: gym.Env,
        skip_frames: int = 50,
        max_negative_steps: int = 100,
        reward_clip: float = 1.0,
    ):
        super().__init__(env)
        self.skip_frames = skip_frames
        self.max_negative_steps = max_negative_steps
        self.reward_clip = reward_clip
        self._negative_step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._negative_step_count = 0

        # Skip the zoom-in animation at episode start
        for _ in range(self.skip_frames):
            obs, _, terminated, truncated, info = self.env.step([0.0, 0.0, 0.0])
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).astype(float).tolist()
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track consecutive negative-reward steps
        if reward < 0:
            self._negative_step_count += 1
        else:
            self._negative_step_count = 0

        # Early termination if agent is stuck / off-track
        if self._negative_step_count >= self.max_negative_steps:
            terminated = True

        # Clip reward for stability
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        return obs, reward, terminated, truncated, info


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat each action for N consecutive frames and sum the rewards.
    Reduces effective decision frequency → agent acts on higher-level signals.

    Args:
        env:          The wrapped environment.
        n_repeat:     Number of frames to repeat each action (default: 4).
    """

    def __init__(self, env: gym.Env, n_repeat: int = 4):
        super().__init__(env)
        self.n_repeat = n_repeat

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).astype(float).tolist()
        total_reward = 0.0
        for _ in range(self.n_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Normalise pixel observations from [0, 255] → [0, 1].

    Note: SB3's CnnPolicy does this internally when normalize_images=True,
    so you typically DON'T need this wrapper. Included for reference.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class RewardLogger(gym.Wrapper):
    """
    Print episode stats to stdout. Useful for quick debugging without TensorBoard.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_count = 0

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        self._ep_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += reward
        self._ep_steps += 1

        if terminated or truncated:
            self._ep_count += 1
            print(
                f"[Episode {self._ep_count:4d}] "
                f"reward={self._ep_reward:7.2f}  steps={self._ep_steps:4d}"
            )

        return obs, reward, terminated, truncated, info
