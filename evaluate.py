"""
evaluate.py — Load & Evaluate a Trained PPO Agent
===================================================
Loads a saved model and runs it in the CarRacing environment.
Can render to screen, record video, or run silent benchmarks.

Usage:
    python evaluate.py                                     # use best model
    python evaluate.py --model models/best/best_model     # specific model
    python evaluate.py --episodes 20 --record             # record 20 episodes
    python evaluate.py --render                            # watch live
"""

import os
import argparse
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from wrappers import CarRacingWrapper


def make_eval_env(render_mode=None, record=False, video_dir="videos/eval"):
    """Build a single evaluation environment."""

    def _init():
        mode = render_mode or ("rgb_array" if record else None)
        env = gym.make("CarRacing-v2", render_mode=mode, continuous=True)
        env = CarRacingWrapper(env, skip_frames=50, max_negative_steps=150)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)
        env = Monitor(env)
        if record:
            os.makedirs(video_dir, exist_ok=True)
            env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda e: True,  # record every episode
                name_prefix="car_racing_eval",
            )
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def run_evaluation(model_path: str, n_episodes: int, render: bool, record: bool, video_dir: str):
    """Load model and run evaluation episodes."""

    print(f"\nLoading model: {model_path}")
    render_mode = "human" if render else None
    env = make_eval_env(render_mode=render_mode, record=record, video_dir=video_dir)

    model = PPO.load(model_path, env=env, device="auto")
    print(f"Model loaded. Running {n_episodes} evaluation episodes...\n")

    # Use SB3's built-in evaluation function
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        render=render,
        return_episode_rewards=False,
    )

    print(f"\n{'─'*40}")
    print(f"  Episodes:       {n_episodes}")
    print(f"  Mean reward:    {mean_reward:.2f}")
    print(f"  Std  reward:    {std_reward:.2f}")
    print(f"  Score range:    [{mean_reward - std_reward:.2f}, {mean_reward + std_reward:.2f}]")
    print(f"{'─'*40}")

    # CarRacing scoring guide:
    #   < 0    : agent is stuck / going backwards
    #   100–300: agent learns to stay on track
    #   500–700: agent drives competently around most of the track
    #   > 900  : near-human performance (considered "solved")
    score = mean_reward
    if score < 100:
        verdict = "Still learning — keep training!"
    elif score < 400:
        verdict = "Basic driving — shows progress."
    elif score < 700:
        verdict = "Competent driver — solid result."
    elif score < 900:
        verdict = "Excellent! Near expert performance."
    else:
        verdict = "Solved! 900+ is human-level."

    print(f"\n  Verdict: {verdict}")

    if record:
        print(f"\n  Videos saved to: {video_dir}/")

    env.close()
    return mean_reward, std_reward


def run_manual_rollout(model_path: str, n_episodes: int, render: bool, record: bool, video_dir: str):
    """
    Manual rollout loop — more control than evaluate_policy().
    Prints per-episode stats and shows a progress bar.
    """

    render_mode = "human" if render else ("rgb_array" if record else None)
    env = make_eval_env(render_mode=render_mode, record=record, video_dir=video_dir)
    model = PPO.load(model_path, env=env, device="auto")

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_steps += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

        bar = "█" * int(ep_reward / 10) if ep_reward > 0 else ""
        print(f"  Episode {ep+1:3d}/{n_episodes}  |  reward={ep_reward:7.1f}  steps={ep_steps:4d}  {bar}")

    print(f"\n  Mean: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Best: {max(episode_rewards):.2f}   Worst: {min(episode_rewards):.2f}")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO CarRacing agent")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best/best_model",
        help="Path to model .zip (without extension)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--render", action="store_true", help="Render to screen (human mode)")
    parser.add_argument("--record", action="store_true", help="Record episodes to video files")
    parser.add_argument("--video-dir", type=str, default="videos/eval", help="Directory for saved videos")
    parser.add_argument("--verbose", action="store_true", help="Use manual rollout with per-episode output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model + ".zip"):
        print(f"ERROR: Model not found at {args.model}.zip")
        print("Train a model first:  python train.py")
        exit(1)

    if args.verbose:
        run_manual_rollout(
            args.model, args.episodes, args.render, args.record, args.video_dir
        )
    else:
        run_evaluation(
            args.model, args.episodes, args.render, args.record, args.video_dir
        )
