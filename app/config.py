from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "decoder.json"


@dataclass
class DecoderConfig:
    dt: float = 0.020
    window_steps: int = 13
    publish_steps: int = 5
    alpha: float = 0.30
    deadzone: float = 0.05
    turn_sign: float = 1.0
    max_turn_rate: float = 2.2
    fixed_speed: float = 1.0
    action_threshold: float = 0.60
    action_hold_seconds: float = 0.80
    odor_baseline_hz: float = 0.3
    odor_scale_hz: float = 25.5
    odor_turn_gain: float = 2.5
    baseline: dict[str, float] = field(
        default_factory=lambda: {
            "steer_left": 0.0,
            "steer_right": 0.0,
            "escape_left": 0.0,
            "escape_right": 0.0,
        }
    )
    scale: dict[str, float] = field(
        default_factory=lambda: {
            "steer_left": 20.0,
            "steer_right": 20.0,
            "escape_left": 20.0,
            "escape_right": 20.0,
        }
    )
    calibration: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "DecoderConfig":
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
