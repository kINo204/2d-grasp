from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ShapeType = Literal["circle", "box", "polygon"]


@dataclass
class CurriculumScheduler:
    """Simple stage scheduler for shape complexity."""

    enabled: bool = True
    window_size: int = 100
    promote_threshold: float = 0.55
    min_episodes_per_stage: int = 200
    stage: int = 0
    episodes_in_stage: int = 0
    _recent_successes: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self._recent_successes = deque(maxlen=self.window_size)

    def current_stage(self) -> int:
        if not self.enabled:
            return 2
        return self.stage

    def update(self, success: bool) -> None:
        if not self.enabled:
            return
        self.episodes_in_stage += 1
        self._recent_successes.append(1.0 if success else 0.0)
        if self.stage >= 2:
            return
        if self.episodes_in_stage < self.min_episodes_per_stage:
            return
        if len(self._recent_successes) < self.window_size:
            return
        if float(np.mean(self._recent_successes)) >= self.promote_threshold:
            self.stage += 1
            self.episodes_in_stage = 0
            self._recent_successes.clear()

    def sample_shape(self, rng: np.random.Generator) -> ShapeType:
        stage = self.current_stage()
        if stage == 0:
            return "circle"
        if stage == 1:
            return "circle" if rng.random() < 0.3 else "box"
        choices: tuple[ShapeType, ...] = ("circle", "box", "polygon")
        probs = np.array([0.2, 0.4, 0.4], dtype=np.float64)
        return str(rng.choice(choices, p=probs))  # type: ignore[return-value]
