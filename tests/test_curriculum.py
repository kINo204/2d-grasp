from __future__ import annotations

import importlib
import unittest


class TestCurriculum(unittest.TestCase):
    def test_window_size_is_respected(self) -> None:
        curriculum = importlib.import_module("2d_grasp.envs.curriculum")
        scheduler = curriculum.CurriculumScheduler(window_size=7)
        self.assertEqual(scheduler._recent_successes.maxlen, 7)


if __name__ == "__main__":
    unittest.main()
