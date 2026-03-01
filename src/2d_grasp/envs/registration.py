from __future__ import annotations

import gymnasium as gym


def register_envs() -> None:
    if "DexGrasp2D-v0" in gym.registry:
        return
    gym.register(
        id="DexGrasp2D-v0",
        entry_point="2d_grasp.envs.dex_grasp_2d:DexGrasp2DEnv",
        max_episode_steps=300,
    )
