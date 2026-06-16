"""
CarRacing-v2 — PPO Training Script
====================================
Train a PPO agent to race a car using pixel observations.
Uses Stable-Baselines3 with a CNN policy and frame stacking.

Usage:
    python train.py               # train from scratch
    python train.py --resume      # resume from last checkpoint
    python train.py --timesteps 2000000
"""

import os
import argparse
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    RecordVideo,
)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from config import TrainingConfig
from wrappers import CarRacingWrapper


# ─── Environment Factory ───────────────────────────────────────────────────────

def make_env(render_mode=None, record_video=False, video_dir="videos/eval"):
    """Create and wrap a single CarRacing environment."""

    def _init():
        env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=True)

        # Custom wrapper: clip rewards, skip initial zoom frames
        env = CarRacingWrapper(env)

        # Grayscale: reduces 3-channel RGB → 1-channel (faster training)
        env = GrayScaleObservation(env, keep_dim=True)

        # Resize: 96×96 → 84×84 (standard in DRL literature)
        env = ResizeObservation(env, shape=84)

        # Monitor: logs episode rewards/lengths to CSV
        env = Monitor(env)

        if record_video:
            env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: e % 10 == 0)

        return env

    return _init


def build_vec_env(n_envs: int, seed: int = 42):
    """Create a vectorized environment with frame stacking."""
    vec_env = make_vec_env(
        make_env(),
        n_envs=n_envs,
        seed=seed,
    )

    # Stack 4 consecutive frames so agent perceives motion/velocity
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # SB3 CNN policy expects (C, H, W); VecTransposeImage handles this
    vec_env = VecTransposeImage(vec_env)

    return vec_env


# ─── PPO Model ─────────────────────────────────────────────────────────────────

def build_model(env, cfg: TrainingConfig, tensorboard_log: str):
    """Instantiate the PPO model with CNN policy."""
    model = PPO(
        policy="CnnPolicy",
        env=env,
        # Core PPO hyperparameters
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,               # Steps per env before update
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,             # Gradient update passes per rollout
        gamma=cfg.gamma,                   # Discount factor
        gae_lambda=cfg.gae_lambda,         # GAE smoothing
        clip_range=cfg.clip_range,         # PPO clipping epsilon
        clip_range_vf=cfg.clip_range_vf,   # Value function clipping
        ent_coef=cfg.ent_coef,             # Entropy bonus (encourages exploration)
        vf_coef=cfg.vf_coef,               # Value function loss weight
        max_grad_norm=cfg.max_grad_norm,
        # CNN policy kwargs: custom architecture
        policy_kwargs=dict(
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),  # separate actor/critic heads
            normalize_images=True,
        ),
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=42,
        device="auto",  # uses GPU if available, else CPU
    )
    return model


# ─── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks(cfg: TrainingConfig, eval_env):
    """Set up checkpoint + evaluation callbacks."""

    # Save a checkpoint every N steps
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path=cfg.checkpoint_dir,
        name_prefix="ppo_car_racing",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    # Evaluate the agent periodically on a separate env, keep best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg.best_model_dir,
        log_path=cfg.eval_log_dir,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    return CallbackList([checkpoint_cb, eval_cb])


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing-v2")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--model-path", type=str, default=None, help="Path to specific .zip model to resume")
    parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel envs")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainingConfig()

    # Override config with CLI args
    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.n_envs:
        cfg.n_envs = args.n_envs

    # Create output directories
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)
    os.makedirs(cfg.eval_log_dir, exist_ok=True)
    os.makedirs(cfg.tensorboard_log_dir, exist_ok=True)

    run_name = f"ppo_car_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*50}")
    print(f"  Run: {run_name}")
    print(f"  Envs: {cfg.n_envs}  |  Timesteps: {cfg.total_timesteps:,}")
    print(f"  Device: auto (GPU if available)")
    print(f"{'='*50}\n")

    # Build environments
    train_env = build_vec_env(n_envs=cfg.n_envs, seed=42)
    eval_env = build_vec_env(n_envs=1, seed=99)  # separate seed for eval

    # Build or load model
    if args.resume or args.model_path:
        model_path = args.model_path or _find_latest_checkpoint(cfg.checkpoint_dir)
        print(f"Resuming from: {model_path}")
        model = PPO.load(model_path, env=train_env, device="auto")
    else:
        model = build_model(train_env, cfg, tensorboard_log=cfg.tensorboard_log_dir)

    # Print model summary
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy parameters: {total_params:,}")

    # Build callbacks
    callbacks = build_callbacks(cfg, eval_env)

    # Train!
    print("\nStarting training... (Ctrl+C to interrupt and save)\n")
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not args.resume,
            tb_log_name=run_name,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving model...")

    # Save final model
    final_path = os.path.join(cfg.checkpoint_dir, f"{run_name}_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")
    print(f"Best model saved to:  {cfg.best_model_dir}/best_model.zip")
    print(f"\nView TensorBoard:\n  tensorboard --logdir {cfg.tensorboard_log_dir}")

    train_env.close()
    eval_env.close()


def _find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recently saved checkpoint .zip file."""
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".zip") and "ppo_car_racing" in f
    ]
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    files.sort()
    return os.path.join(checkpoint_dir, files[-1][:-4])  # strip .zip


if __name__ == "__main__":
    main()
