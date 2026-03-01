from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import gymnasium as gym


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on DexGrasp2D-v0")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/dexgrasp_ppo"))
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable SB3 progress bar (requires tqdm + rich extras).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    importlib.import_module("2d_grasp")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for training. "
            "Install with: uv add stable-baselines3 torch"
        ) from exc

    args.log_dir.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(
        "DexGrasp2D-v0",
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"shape_curriculum": True},
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.log_dir),
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)
    model_path = args.log_dir / "ppo_dexgrasp.zip"
    stats_path = args.log_dir / "vecnormalize.pkl"
    model.save(str(model_path))
    vec_env.save(str(stats_path))
    vec_env.close()
    print(f"Saved model: {model_path}")
    print(f"Saved normalization stats: {stats_path}")


if __name__ == "__main__":
    main()
