from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import gymnasium as gym
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO on DexGrasp2D-v0")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    importlib.import_module("2d_grasp")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for evaluation. "
            "Install with: uv add stable-baselines3 torch"
        ) from exc

    env_kwargs = {"shape_curriculum": False, "max_steps": args.max_steps}
    vec = DummyVecEnv([lambda: gym.make("DexGrasp2D-v0", **env_kwargs)])
    vec = VecNormalize.load(str(args.stats), vec)
    vec.training = False
    vec.norm_reward = False
    model = PPO.load(str(args.model), env=vec)
    vec.seed(args.seed)

    success = 0
    hold_scores: list[float] = []
    stability_scores: list[float] = []
    drop_count = 0
    for ep in range(args.episodes):
        obs = vec.reset()
        done = False
        ep_stability: list[float] = []
        ep_contacts: list[int] = []
        while not done:
            action, _ = model.predict(obs, deterministic=True) # type: ignore
            obs, _, dones, infos = vec.step(action)
            done = bool(dones[0])
            info = infos[0]
            ep_stability.append(float(info.get("stability_score", 0.0)))
            ep_contacts.append(int(info.get("contact_count", 0)))
            if done:
                if info.get("is_success", False):
                    success += 1
                else:
                    drop_count += 1
        hold_scores.append(float(np.mean(ep_contacts)))
        stability_scores.append(float(np.mean(ep_stability)))

    success_rate = success / args.episodes
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Avg contact count proxy: {np.mean(hold_scores):.3f}")
    print(f"Avg stability score: {np.mean(stability_scores):.3f}")
    print(f"Drop count proxy: {drop_count}")
    vec.close()


if __name__ == "__main__":
    main()
