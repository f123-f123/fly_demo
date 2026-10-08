from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime

import numpy as np
from flybrain import FlyBrain

from .config import DecoderConfig


OUTPUT_SPECS = {
    "steer_left": (["DNa02"], "L"),
    "steer_right": (["DNa02"], "R"),
    "escape_left": (["DNp01"], "L"),
    "escape_right": (["DNp01"], "R"),
}
STIMULUS_SPECS = {
    "target_left": (["LC10a"], "L", 0.8),
    "target_right": (["LC10a"], "R", 0.8),
    "loom_left": (["LC4", "LPLC2"], "L", 0.8),
    "loom_right": (["LC4", "LPLC2"], "R", 0.8),
}


def count_rates(
    brain: FlyBrain,
    outputs: dict[str, np.ndarray],
    steps: int,
    inject=(),
) -> dict[str, np.ndarray]:
    counts = {name: np.zeros(brain.batch, dtype=np.int32) for name in outputs}
    lookup = {name: set(indices.tolist()) for name, indices in outputs.items()}
    for _ in range(steps):
        for batch_index, fired in enumerate(brain.step(inject=inject)):
            fired_set = set(fired.tolist())
            for name, population in lookup.items():
                counts[name][batch_index] += len(fired_set & population)
    seconds = steps * brain.dt
    return {
        name: count / len(outputs[name]) / seconds
        for name, count in counts.items()
    }


def calibrate(device: str, flies: int, seed: int) -> DecoderConfig:
    config = DecoderConfig.load()
    brain = FlyBrain(
        device=device,
        batch=flies,
        seed=seed,
        dt=config.dt,
        sensory_input=False,
    )
    outputs = {
        name: brain.cells(types, side=side)
        for name, (types, side) in OUTPUT_SPECS.items()
    }
    if any(len(indices) == 0 for indices in outputs.values()):
        raise RuntimeError("One or more calibration output populations are empty")

    steps = round(1.0 / config.dt)
    brain.reset(seed)
    count_rates(brain, outputs, steps)
    baseline_samples = count_rates(brain, outputs, steps * 2)
    baseline = {name: float(values.mean()) for name, values in baseline_samples.items()}

    responses: dict[str, dict[str, list[float]]] = {}
    for index, (condition, (types, side, strength)) in enumerate(STIMULUS_SPECS.items()):
        brain.reset(seed + 100 + index)
        count_rates(brain, outputs, steps)
        target = brain.cells(types, side=side)
        rates = count_rates(brain, outputs, steps, inject=[(target, strength)])
        responses[condition] = {
            name: values.round(4).tolist() for name, values in rates.items()
        }

    scale_pairs = {
        "steer_left": "target_left",
        "steer_right": "target_right",
        "escape_left": "loom_left",
        "escape_right": "loom_right",
    }
    scale = {}
    for output, condition in scale_pairs.items():
        values = np.asarray(responses[condition][output])
        scale[output] = max(baseline[output] + 5.0, float(np.percentile(values, 90)))

    target_left = responses["target_left"]
    target_right = responses["target_right"]
    left_difference = np.mean(target_left["steer_right"]) - np.mean(target_left["steer_left"])
    right_difference = np.mean(target_right["steer_right"]) - np.mean(target_right["steer_left"])
    turn_sign = 1.0 if right_difference > left_difference else -1.0

    config.baseline = {name: round(value, 4) for name, value in baseline.items()}
    config.scale = {name: round(value, 4) for name, value in scale.items()}
    config.turn_sign = turn_sign
    config.calibration = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": brain.device,
        "flies": flies,
        "seed": seed,
        "stimulus_strength": 0.8,
        "responses_hz": responses,
    }
    config.save()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate FlyBrain control populations")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--flies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    config = calibrate(args.device, args.flies, args.seed)
    print("Calibration saved to config/decoder.json")
    print(f"turn_sign={config.turn_sign:+.0f}")
    print(f"baseline={config.baseline}")
    print(f"scale={config.scale}")


if __name__ == "__main__":
    main()

