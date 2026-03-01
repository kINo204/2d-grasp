from __future__ import annotations

import importlib
import unittest

import gymnasium as gym
import numpy as np


class TestDexGraspEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        importlib.import_module("2d_grasp")
        reward_module = importlib.import_module("2d_grasp.envs.reward")
        cls.ContactFeature = reward_module.ContactFeature
        cls.compute_reward = staticmethod(reward_module.compute_reward)

    def test_registration_and_api(self) -> None:
        env = gym.make("DexGrasp2D-v0")
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIn("curriculum_stage", info)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("reward_terms", info)
        env.close()

    def test_seed_determinism(self) -> None:
        env1 = gym.make("DexGrasp2D-v0")
        env2 = gym.make("DexGrasp2D-v0")
        obs1, _ = env1.reset(seed=7)
        obs2, _ = env2.reset(seed=7)
        np.testing.assert_allclose(obs1, obs2, atol=1e-6)
        env1.close()
        env2.close()

    def test_reward_contact_monotonicity(self) -> None:
        zero_contacts = [
            self.ContactFeature(
                False, np.zeros(2), np.array([0.0, 1.0]), 0.0, 0.0, 0.0
            )
            for _ in range(3)
        ]
        two_contacts = [
            self.ContactFeature(
                True, np.zeros(2), np.array([0.0, 1.0]), 0.8, 10.0, 0.1
            ),
            self.ContactFeature(
                True, np.zeros(2), np.array([1.0, 0.0]), 0.7, 12.0, 0.1
            ),
            self.ContactFeature(
                False, np.zeros(2), np.array([0.0, 1.0]), 0.0, 0.0, 0.0
            ),
        ]
        weights = {
            "w_contact": 0.25,
            "w_inward": 0.20,
            "w_wrench": 0.25,
            "w_still": 0.15,
            "w_lift": 0.15,
            "w_reg": 0.05,
        }
        r0 = self.compute_reward(
            zero_contacts,
            object_pos=np.zeros(2),
            object_lin_vel=np.zeros(2),
            object_ang_vel=0.0,
            lift_progress=0.0,
            action=np.zeros(3),
            prev_action=np.zeros(3),
            reward_weights=weights,
            success=False,
            dropped=False,
        )
        r2 = self.compute_reward(
            two_contacts,
            object_pos=np.zeros(2),
            object_lin_vel=np.zeros(2),
            object_ang_vel=0.0,
            lift_progress=0.0,
            action=np.zeros(3),
            prev_action=np.zeros(3),
            reward_weights=weights,
            success=False,
            dropped=False,
        )
        self.assertGreater(r2.contact, r0.contact)

    def test_rgb_array_render(self) -> None:
        env = gym.make("DexGrasp2D-v0", render_mode="rgb_array")
        env.reset(seed=2)
        frame = env.render()
        self.assertIsInstance(frame, np.ndarray)
        assert frame is not None
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.dtype, np.uint8)
        env.step(env.action_space.sample())
        frame2 = env.render()
        self.assertEqual(frame2.shape, frame.shape)
        env.close()

    def test_reachability_reset_places_object_in_range(self) -> None:
        env = gym.make("DexGrasp2D-v0", reachability_reset=True)
        env.reset(seed=3)
        core = env.unwrapped
        obj = core._object
        self.assertIsNotNone(obj)
        x = float(obj.position[0])
        y = float(obj.position[1])
        l1, l2 = core.config.finger_link_lengths
        r_max = (l1 + l2) * 0.95 + 1e-6
        dists = [
            np.hypot(x - ax, y - core.config.palm_height)
            for ax in core.config.finger_base_x
        ]
        self.assertLessEqual(min(dists), r_max)
        env.close()


if __name__ == "__main__":
    unittest.main()
