from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DecoderConfig


@dataclass
class DecodeResult:
    speed: float
    raw_turn: float
    filtered_turn: float
    action: bool
    normalized: dict[str, float]
    odor_turn: float = 0.0


class RuleDecoder:
    def __init__(self, config: DecoderConfig):
        self.config = config
        self.turn_state = 0.0
        self.action_remaining = 0.0

    def reset(self) -> None:
        self.turn_state = 0.0
        self.action_remaining = 0.0

    def normalize(self, name: str, value: float) -> float:
        low = self.config.baseline.get(name, self.config.odor_baseline_hz)
        high = self.config.scale.get(name, self.config.odor_scale_hz)
        return float(np.clip((value - low) / max(high - low, 1e-6), 0.0, 1.0))

    def decode(
        self,
        rates: dict[str, float],
        elapsed_seconds: float,
    ) -> DecodeResult:
        normalized = {name: self.normalize(name, rates[name]) for name in rates}
        for name in ("odor_left", "odor_right"):
            if name not in normalized:
                normalized[name] = self.normalize(name, self.config.odor_baseline_hz)
        neural_turn = self.config.turn_sign * (
            normalized["steer_right"] - normalized["steer_left"]
        )
        odor_turn = self.config.odor_turn_gain * (
            normalized["odor_right"] - normalized["odor_left"]
        )
        raw_turn = neural_turn + odor_turn
        self.turn_state += self.config.alpha * (raw_turn - self.turn_state)
        if abs(self.turn_state) < self.config.deadzone:
            self.turn_state = 0.0

        escape_level = max(normalized["escape_left"], normalized["escape_right"])
        if escape_level >= self.config.action_threshold:
            self.action_remaining = self.config.action_hold_seconds
        else:
            self.action_remaining = max(0.0, self.action_remaining - elapsed_seconds)

        action = self.action_remaining > 0.0
        speed = self.config.fixed_speed if not action else 0.0
        return DecodeResult(
            speed=speed,
            raw_turn=float(np.clip(raw_turn, -1.0, 1.0)),
            filtered_turn=float(np.clip(self.turn_state, -1.0, 1.0)),
            action=action,
            normalized=normalized,
            odor_turn=float(np.clip(odor_turn, -1.0, 1.0)),
        )
