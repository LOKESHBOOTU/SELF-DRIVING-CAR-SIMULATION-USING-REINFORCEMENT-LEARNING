"""
config.py — Training Hyperparameters & Paths
=============================================
All tunable settings live here. Edit this file rather than
digging through train.py to tweak a hyperparameter.

PPO Hyperparameter guide
─────────────────────────
learning_rate   Too high → unstable training. Too low → slow. 3e-4 is a good default.
n_steps         Rollout length per env. Larger = more stable gradients, more memory.
batch_size      Must divide (n_steps × n_envs). Smaller = noisier, faster updates.
n_epochs        Passes over each rollout. 10 is standard for PPO.
gamma           Discount. 0.99 = values future rewards highly.
gae_lambda      Bias-variance trade-off in advantage estimation. 0.95 is standard.
clip_range      PPO "clipping" epsilon. 0.2 is standard. Lower = more conservative.
ent_coef        Entropy bonus weight. Higher = more exploration. Tune if agent is stuck.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:

    # ── Parallelism ──────────────────────────────────────────────────────────
    n_envs: int = 8                     # parallel environments (use 4-16)

    # ── Training Duration ────────────────────────────────────────────────────
    total_timesteps: int = 2_000_000    # ~1-2M is enough to see competent driving

    # ── PPO Core Hyperparameters ─────────────────────────────────────────────
    learning_rate: float = 3e-4
    n_steps: int = 512                  # steps per env per rollout
    batch_size: int = 128               # must divide n_steps * n_envs
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = None         # set to same as clip_range to also clip VF
    ent_coef: float = 0.01              # entropy bonus — tune up if agent stalls
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_freq: int = 50_000             # evaluate every N steps (across all envs)
    n_eval_episodes: int = 5
    checkpoint_freq: int = 100_000      # save checkpoint every N steps

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    best_model_dir: str = "models/best"
    eval_log_dir: str = "logs/eval"
    tensorboard_log_dir: str = "logs/tensorboard"
    video_dir: str = "videos"

    def validate(self):
        """Sanity-check hyperparameter relationships."""
        total_rollout = self.n_steps * self.n_envs
        assert total_rollout % self.batch_size == 0, (
            f"batch_size ({self.batch_size}) must divide "
            f"n_steps × n_envs ({total_rollout}). "
            f"Try batch_size={total_rollout // 8}."
        )
        return self


# ── Hyperparameter Presets ──────────────────────────────────────────────────────
# Use these as starting points depending on your hardware.

PRESETS = {

    # Good for CPU-only machines (fewer parallel envs, smaller batches)
    "cpu": TrainingConfig(
        n_envs=4,
        total_timesteps=1_000_000,
        n_steps=256,
        batch_size=64,
    ),

    # Balanced for a mid-range GPU (RTX 3060 / 4060)
    "gpu_medium": TrainingConfig(
        n_envs=8,
        total_timesteps=2_000_000,
        n_steps=512,
        batch_size=128,
    ),

    # Maximum throughput for high-end GPU (RTX 3090 / A100)
    "gpu_high": TrainingConfig(
        n_envs=16,
        total_timesteps=5_000_000,
        n_steps=1024,
        batch_size=256,
        ent_coef=0.005,
    ),

    # Quick smoke test — verifies everything runs without errors
    "smoke_test": TrainingConfig(
        n_envs=2,
        total_timesteps=10_000,
        n_steps=64,
        batch_size=32,
        eval_freq=5_000,
        checkpoint_freq=5_000,
        n_eval_episodes=1,
    ),
}
