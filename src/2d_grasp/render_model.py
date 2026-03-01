from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import gymnasium as gym
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a trained PPO model in DexGrasp2D-v0"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    importlib.import_module("2d_grasp")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required. Install with: uv add stable-baselines3 torch"
        ) from exc

    env_kwargs = {
        "shape_curriculum": False,
        "max_steps": args.max_steps,
        "render_mode": args.render_mode,
    }
    vec = DummyVecEnv([lambda: gym.make("DexGrasp2D-v0", **env_kwargs)])
    vec = VecNormalize.load(str(args.stats), vec)
    vec.training = False
    vec.norm_reward = False
    vec.seed(args.seed)
    model = PPO.load(str(args.model), env=vec)

    success_count = 0
    for ep in range(args.episodes):
        obs = vec.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        last_info: dict[str, object] = {}
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)  # type: ignore[arg-type]
            obs, rewards, dones, infos = vec.step(action)
            ep_reward += float(rewards[0])
            done = bool(dones[0])
            last_info = infos[0]
            ep_steps += 1
            if args.render_mode == "rgb_array":
                frame = vec.render()
                if isinstance(frame, np.ndarray) and ep_steps == 1:
                    print(f"rgb_array frame: {frame.shape}, {frame.dtype}")
        is_success = bool(last_info.get("is_success", False))
        success_count += int(is_success)
        print(
            f"Episode {ep + 1}/{args.episodes} | steps={ep_steps} | "
            f"reward={ep_reward:.3f} | success={is_success}"
        )

    print(f"Success rate: {success_count / max(args.episodes, 1):.3f}")
    vec.close()


if __name__ == "__main__":
    main()
