"""2D grasping environments."""

from .envs.registration import register_envs

register_envs()

__all__ = ["register_envs"]
