from __future__ import annotations

from collections import deque

import numpy as np


class ActivityWindow:
    def __init__(
        self,
        populations: dict[str, np.ndarray],
        dt: float,
        window_steps: int,
    ):
        empty = [name for name, indices in populations.items() if len(indices) == 0]
        if empty:
            raise ValueError(f"Empty output populations: {', '.join(empty)}")
        self.populations = populations
        self.dt = dt
        self.window_steps = window_steps
        self.buffers = {
            name: deque(maxlen=window_steps) for name in populations
        }
        self.last_counts = {name: 0 for name in populations}

    def reset(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()
        self.last_counts = {name: 0 for name in self.populations}

    def push(self, fired: np.ndarray) -> dict[str, float]:
        for name, population in self.populations.items():
            count = int(np.isin(fired, population).sum())
            self.last_counts[name] = count
            self.buffers[name].append(count)

        return {
            name: sum(self.buffers[name])
            / len(population)
            / (len(self.buffers[name]) * self.dt)
            for name, population in self.populations.items()
        }

