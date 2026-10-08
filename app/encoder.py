from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .world import Sensors


@dataclass
class Stimulus:
    lc10a_left: float = 0.0
    lc10a_right: float = 0.0
    loom_left: float = 0.0
    loom_right: float = 0.0
    odor_left: float = 0.0
    odor_right: float = 0.0
    source: str = "world"

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class FeatureEncoder:
    def __init__(
        self,
        target_strength: float = 0.8,
        looming_gain: float = 2.2,
        odor_gain: float = 5.0,
    ):
        self.target_strength = target_strength
        self.looming_gain = looming_gain
        self.odor_gain = odor_gain

    def encode(self, sensors: Sensors) -> Stimulus:
        result = Stimulus()
        if sensors.target_visible:
            side = sensors.target_bearing_normalized
            size = float(np.clip(sensors.target_angular_size / math.radians(5.0), 0.0, 1.0))
            motion = 0.7 + 0.3 * float(np.clip(abs(sensors.target_angular_rate) * 2.0, 0.0, 1.0))
            strength = self.target_strength * sensors.target_contrast * size * motion
            result.lc10a_left = strength * (1.0 - side) / 2.0
            result.lc10a_right = strength * (1.0 + side) / 2.0

        if sensors.obstacle_visible:
            strength = float(np.clip(sensors.looming_rate * self.looming_gain, 0.0, 0.8))
            side = float(np.clip(sensors.obstacle_bearing / math.radians(135.0), -1.0, 1.0))
            result.loom_left = float(np.clip(strength * (1.0 - 0.75 * side), 0.0, 0.8))
            result.loom_right = float(np.clip(strength * (1.0 + 0.75 * side), 0.0, 0.8))
        result.odor_left = float(np.clip(sensors.odor_left * self.odor_gain, 0.0, 0.8))
        result.odor_right = float(np.clip(sensors.odor_right * self.odor_gain, 0.0, 0.8))
        return result
