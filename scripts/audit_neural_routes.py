from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "baseline": None,
    "odor_left": "odor_left",
    "odor_right": "odor_right",
}


def rate_hz(
    spike_trains: list[list[np.ndarray]],
    population: np.ndarray,
    dt: float,
) -> float:
    spikes = sum(
        int(np.isin(fired, population).sum())
        for trial in spike_trains
        for fired in trial
    )
    trials = len(spike_trains)
    steps = len(spike_trains[0])
    return spikes / max(len(population) * trials * steps * dt, 1e-9)


def run(device: str, batch: int, seed: int) -> dict[str, object]:
    brain = FlyBrain(
        device=device,
        batch=batch,
        seed=seed,
        sensory_input=False,
        dt=0.02,
    )
    inputs = {
        "odor_left": np.concatenate([
            brain.cells(["ORN_DM1"], side="L"),
            brain.cells(["ORN_VA2"], side="L"),
        ]),
        "odor_right": np.concatenate([
            brain.cells(["ORN_DM1"], side="R"),
            brain.cells(["ORN_VA2"], side="R"),
        ]),
    }
    outputs = {
        "ORN_food_L": inputs["odor_left"],
        "ORN_food_R": inputs["odor_right"],
        "DM1_lPN_L": brain.cells(["DM1_lPN"], side="L"),
        "DM1_lPN_R": brain.cells(["DM1_lPN"], side="R"),
        "VA2_adPN_L": brain.cells(["VA2_adPN"], side="L"),
        "VA2_adPN_R": brain.cells(["VA2_adPN"], side="R"),
        "DNa02_L": brain.cells(["DNa02"], side="L"),
        "DNa02_R": brain.cells(["DNa02"], side="R"),
        "DNp01_L": brain.cells(["DNp01"], side="L"),
        "DNp01_R": brain.cells(["DNp01"], side="R"),
    }
    for cell_type in ("DNg01", "DNg02_a", "DNg02_b", "DNg02_c", "DNg03", "DNg05_a", "DNg05_b", "DNg06_a", "DNg06_b,DNg06_c"):
        population = brain.cells([cell_type])
        if len(population):
            outputs[cell_type] = population

    results: dict[str, dict[str, float]] = {}
    for condition, input_name in CONDITIONS.items():
        brain.reset(seed)
        for _ in range(50):
            brain.step()
        spike_trains: list[list[np.ndarray]] = [[] for _ in range(batch)]
        for _ in range(60):
            inject = () if input_name is None else ((inputs[input_name], 0.65),)
            fired_by_trial = brain.step(inject=inject)
            for trial, fired in enumerate(fired_by_trial):
                spike_trains[trial].append(fired)
        results[condition] = {
            name: round(rate_hz(spike_trains, population, brain.dt), 4)
            for name, population in outputs.items()
        }

    baseline = results["baseline"]
    deltas = {
        condition: {
            name: round(rate - baseline[name], 4)
            for name, rate in values.items()
        }
        for condition, values in results.items()
        if condition != "baseline"
    }
    return {
        "device": brain.device,
        "batch": batch,
        "seed": seed,
        "dt": brain.dt,
        "stimulus": {
            "types": ["ORN_DM1", "ORN_VA2"],
            "amount": 0.65,
            "input_sizes": {name: len(indices) for name, indices in inputs.items()},
        },
        "rates_hz": results,
        "delta_hz": deltas,
        "vertical_control_status": "unverified_height_hold_policy",
        "vertical_candidates": ["DNg01", "DNg02", "DNg03", "DNg05", "DNg06"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit olfactory and flight-DN routes")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", type=Path, default=ROOT / "config" / "neural_routes.json")
    args = parser.parse_args()
    report = run(args.device, args.batch, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
