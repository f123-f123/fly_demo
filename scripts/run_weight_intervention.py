from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from flybrain import FlyBrain
from flybrain.data import DATA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DecoderConfig
from app.decoder import RuleDecoder
from app.interventions import OutgoingWeightScaler


CONDITIONS = (
    ("baseline", None),
    ("sham", 1.0),
    ("0x", 0.0),
    ("0.5x", 0.5),
    ("2x", 2.0),
)
SIDES = ("left", "right")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batches(fired, batch: int) -> list[np.ndarray]:
    return fired if isinstance(fired, list) else [fired]


def population_counts(fired_by_fly: list[np.ndarray], population: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(np.isin(fired, population).sum()) for fired in fired_by_fly
    ], dtype=np.int32)


def summarize(values: list[float], digits: int = 6) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), digits),
        "std": round(float(array.std()), digits),
        "min": round(float(array.min()), digits),
        "max": round(float(array.max()), digits),
        "values": np.round(array, digits).tolist(),
    }


def run_block(
    brain: FlyBrain,
    config: DecoderConfig,
    populations: dict[str, np.ndarray],
    stimulus: np.ndarray,
    expected_turn_sign: int,
    seed: int,
    warmup_steps: int,
    stimulus_steps: int,
    stimulus_strength: float,
) -> dict[str, list[float]]:
    brain.reset(seed)
    for _ in range(warmup_steps):
        brain.step()

    output_names = ("steer_left", "steer_right", "escape_left", "escape_right")
    total_counts = {
        name: np.zeros(brain.batch, dtype=np.int32) for name in (*output_names, "stimulus")
    }
    rate_windows = {
        name: deque(maxlen=config.window_steps) for name in output_names
    }
    decoders = [RuleDecoder(config) for _ in range(brain.batch)]
    integrated_turn = np.zeros(brain.batch, dtype=float)
    final_turn = np.zeros(brain.batch, dtype=float)
    action_controls = np.zeros(brain.batch, dtype=np.int32)
    total_spikes = np.zeros(brain.batch, dtype=np.int64)
    peak_fraction = np.zeros(brain.batch, dtype=float)

    for step in range(stimulus_steps):
        fired_by_fly = batches(
            brain.step(inject=((stimulus, stimulus_strength),)),
            brain.batch,
        )
        for batch_index, fired in enumerate(fired_by_fly):
            count = len(fired)
            total_spikes[batch_index] += count
            peak_fraction[batch_index] = max(
                peak_fraction[batch_index], count / brain.n
            )

        for name in output_names:
            counts = population_counts(fired_by_fly, populations[name])
            total_counts[name] += counts
            rate_windows[name].append(counts)
        total_counts["stimulus"] += population_counts(fired_by_fly, stimulus)

        if (step + 1) % config.publish_steps != 0:
            continue
        elapsed = config.publish_steps * config.dt
        rates = {
            name: np.sum(np.stack(window), axis=0)
            / len(populations[name])
            / (len(window) * config.dt)
            for name, window in rate_windows.items()
        }
        for batch_index, decoder in enumerate(decoders):
            decoded = decoder.decode(
                {name: float(rate[batch_index]) for name, rate in rates.items()},
                elapsed_seconds=elapsed,
            )
            final_turn[batch_index] = decoded.filtered_turn
            integrated_turn[batch_index] += decoded.filtered_turn * config.max_turn_rate * elapsed
            action_controls[batch_index] += int(decoded.action)

    seconds = stimulus_steps * config.dt
    average_rates = {
        name: total_counts[name] / len(populations[name]) / seconds
        for name in output_names
    }
    stimulus_rates = total_counts["stimulus"] / len(stimulus) / seconds
    mean_network_rates = total_spikes / brain.n / seconds
    direction_correct = (final_turn * expected_turn_sign) > 0.0
    return {
        **{name: values.tolist() for name, values in average_rates.items()},
        "stimulus_rate_hz": stimulus_rates.tolist(),
        "final_turn": final_turn.tolist(),
        "integrated_turn_radians": integrated_turn.tolist(),
        "direction_correct": direction_correct.astype(float).tolist(),
        "action_control_fraction": (action_controls / max(stimulus_steps // config.publish_steps, 1)).tolist(),
        "whole_brain_spikes": total_spikes.astype(float).tolist(),
        "whole_brain_rate_hz_per_neuron": mean_network_rates.tolist(),
        "peak_fraction_spiking_per_step": peak_fraction.tolist(),
    }


def merge_blocks(blocks: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    keys = blocks[0]
    return {key: sum((block[key] for block in blocks), []) for key in keys}


def summarize_side(raw: dict[str, list[float]]) -> dict[str, object]:
    return {key: summarize(values) for key, values in raw.items()}


def condition_score(condition: dict[str, object], metric: str) -> float:
    values = [
        float(condition["sides"][side][metric]["mean"])
        for side in SIDES
    ]
    return float(np.mean(values))


def ipsilateral_rate(condition: dict[str, object]) -> float:
    return float(np.mean([
        condition["sides"]["left"]["steer_left"]["mean"],
        condition["sides"]["right"]["steer_right"]["mean"],
    ]))


def turn_magnitude(condition: dict[str, object]) -> float:
    return float(np.mean([
        abs(condition["sides"][side]["final_turn"]["mean"])
        for side in SIDES
    ]))


def run_experiment(
    device: str,
    batch: int,
    seeds: list[int],
    output: Path,
) -> dict[str, object]:
    config = DecoderConfig.load()
    weights_path = Path(DATA) / "weights.npz"
    brain_path = Path(DATA) / "brain.npz"
    decoder_path = ROOT / "config" / "decoder.json"
    weights_checksum_before = sha256_file(weights_path)
    decoder_checksum = sha256_file(decoder_path)

    brain = FlyBrain(
        device=device,
        batch=batch,
        seed=seeds[0],
        dt=config.dt,
        sensory_input=False,
    )
    scaler = OutgoingWeightScaler(brain)
    lc10a = brain.cells(["LC10a"])
    stimuli = {
        side: brain.cells(["LC10a"], side=side[0].upper())
        for side in SIDES
    }
    populations = {
        "steer_left": brain.cells(["DNa02"], side="L"),
        "steer_right": brain.cells(["DNa02"], side="R"),
        "escape_left": brain.cells(["DNp01"], side="L"),
        "escape_right": brain.cells(["DNp01"], side="R"),
    }
    empty = [name for name, population in {**stimuli, **populations}.items() if len(population) == 0]
    if empty:
        raise RuntimeError(f"Required populations not found: {', '.join(empty)}")

    warmup_steps = round(1.0 / config.dt)
    stimulus_steps = round(1.0 / config.dt)
    strength = 0.8
    condition_results: dict[str, dict[str, object]] = {}
    for label, factor in CONDITIONS:
        started = time.perf_counter()
        if factor is None:
            scaler.restore()
            intervention = {
                "applied": False,
                "factor": 1.0,
                "presynaptic_neurons": len(lc10a),
                "affected_connections": 0,
            }
        else:
            stats = scaler.apply(lc10a, factor)
            intervention = {"applied": True, **stats.to_dict()}

        sides = {}
        for side in SIDES:
            blocks = [
                run_block(
                    brain=brain,
                    config=config,
                    populations=populations,
                    stimulus=stimuli[side],
                    expected_turn_sign=-1 if side == "left" else 1,
                    seed=seed,
                    warmup_steps=warmup_steps,
                    stimulus_steps=stimulus_steps,
                    stimulus_strength=strength,
                )
                for seed in seeds
            ]
            sides[side] = summarize_side(merge_blocks(blocks))

        wall_seconds = time.perf_counter() - started
        simulated_seconds = len(SIDES) * len(seeds) * (warmup_steps + stimulus_steps) * config.dt
        condition_results[label] = {
            "intervention": intervention,
            "sides": sides,
            "performance": {
                "wall_seconds": round(wall_seconds, 4),
                "simulated_seconds_per_fly": round(simulated_seconds, 4),
                "realtime_factor": round(simulated_seconds / wall_seconds, 4),
                "parallel_fly_seconds_per_wall_second": round(simulated_seconds * batch / wall_seconds, 4),
            },
        }

    weights_checksum_after = sha256_file(weights_path)
    if weights_checksum_after != weights_checksum_before:
        raise RuntimeError("Original weights.npz changed during the experiment")

    baseline_sides = condition_results["baseline"]["sides"]
    sham_sides = condition_results["sham"]["sides"]
    dose_labels = ("0x", "0.5x", "baseline", "2x")
    dose_rates = [ipsilateral_rate(condition_results[label]) for label in dose_labels]
    dose_turns = [turn_magnitude(condition_results[label]) for label in dose_labels]
    max_peak_fraction = max(
        condition_results["2x"]["sides"][side]["peak_fraction_spiking_per_step"]["max"]
        for side in SIDES
    )
    baseline_accuracy = condition_score(condition_results["baseline"], "direction_correct")
    sham_accuracy = condition_score(condition_results["sham"], "direction_correct")
    sham_DNa02_difference = abs(
        ipsilateral_rate(condition_results["baseline"])
        - ipsilateral_rate(condition_results["sham"])
    )
    sham_turn_difference = abs(
        turn_magnitude(condition_results["baseline"])
        - turn_magnitude(condition_results["sham"])
    )
    sham_equivalent = (
        sham_DNa02_difference <= 0.25
        and sham_turn_difference <= 0.02
        and abs(baseline_accuracy - sham_accuracy) <= 0.05
    )
    baseline_reproducible = baseline_accuracy >= 0.95
    causal_effect_detected = dose_rates[2] - dose_rates[0] >= 1.0
    DNa02_monotonic = all(a <= b + 1e-9 for a, b in zip(dose_rates, dose_rates[1:]))
    turn_monotonic = all(a <= b + 1e-9 for a, b in zip(dose_turns, dose_turns[1:]))
    network_stable = bool(max_peak_fraction < 0.10)
    report = {
        "schema_version": 1,
        "experiment": "LC10a outgoing-weight intervention to DNa02 steering",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provenance": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "flybrain_version": importlib.metadata.version("flybrain"),
            "device": brain.device,
            "weights_path": str(weights_path),
            "weights_sha256_before": weights_checksum_before,
            "weights_sha256_after": weights_checksum_after,
            "brain_sha256": sha256_file(brain_path),
            "decoder_path": str(decoder_path),
            "decoder_sha256": decoder_checksum,
            "decoder_parameters": asdict(config),
        },
        "protocol": {
            "conditions": [label for label, _ in CONDITIONS],
            "intervention": "bilateral LC10a outgoing synapses, scaled in memory from one immutable base copy",
            "target_trials": list(SIDES),
            "seeds": seeds,
            "parallel_flies_per_seed": batch,
            "trials_per_side_per_condition": len(seeds) * batch,
            "dt_seconds": config.dt,
            "warmup_steps": warmup_steps,
            "stimulus_steps": stimulus_steps,
            "stimulus_strength": strength,
            "decoder_window_steps": config.window_steps,
            "decoder_publish_steps": config.publish_steps,
        },
        "population_sizes": {
            "LC10a_all": len(lc10a),
            **{f"LC10a_{side}": len(population) for side, population in stimuli.items()},
            **{name: len(population) for name, population in populations.items()},
            "whole_brain": brain.n,
        },
        "conditions": condition_results,
        "quality_checks": {
            "original_weights_unchanged": weights_checksum_before == weights_checksum_after,
            "sham_exactly_matches_baseline": baseline_sides == sham_sides,
            "sham_equivalence_tolerance": {
                "ipsilateral_DNa02_mean_hz": 0.25,
                "turn_magnitude_mean": 0.02,
                "direction_accuracy": 0.05,
            },
            "sham_ipsilateral_DNa02_difference_hz": round(sham_DNa02_difference, 6),
            "sham_turn_magnitude_difference": round(sham_turn_difference, 6),
            "sham_equivalent_to_baseline": sham_equivalent,
            "baseline_direction_accuracy": round(baseline_accuracy, 6),
            "sham_direction_accuracy": round(sham_accuracy, 6),
            "baseline_left_and_right_reproducible": baseline_reproducible,
            "ipsilateral_DNa02_rate_by_dose_hz": dict(zip(dose_labels, np.round(dose_rates, 6))),
            "turn_magnitude_by_dose": dict(zip(dose_labels, np.round(dose_turns, 6))),
            "DNa02_rate_is_monotonic": DNa02_monotonic,
            "turn_magnitude_is_monotonic": turn_monotonic,
            "zero_vs_baseline_DNa02_effect_hz": round(dose_rates[2] - dose_rates[0], 6),
            "causal_effect_detected": causal_effect_detected,
            "double_weight_peak_fraction_spiking": round(float(max_peak_fraction), 8),
            "double_weight_network_not_globally_saturated": network_stable,
            "experiment_passed": bool(
                weights_checksum_before == weights_checksum_after
                and sham_equivalent
                and baseline_reproducible
                and DNa02_monotonic
                and turn_monotonic
                and causal_effect_detected
                and network_stable
            ),
        },
        "interpretation_limits": [
            "The intervention scales every outgoing LC10a synapse, not a single direct LC10a-to-DNa02 edge.",
            "Integrated turn is the current Decoder command integrated over time, not a direct measurement of biological muscle output.",
            "CUDA sparse propagation can differ slightly between identical 1x runs, so sham equivalence uses explicit aggregate tolerances rather than bitwise equality.",
            "A causal dose response in this LIF connectome model does not by itself establish biological effect size.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five-condition LC10a weight intervention")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seeds", default="1101,2203,3307")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "lc10a_weight_intervention.json",
    )
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if args.batch < 1 or not seeds:
        parser.error("batch must be positive and at least one seed is required")

    report = run_experiment(args.device, args.batch, seeds, args.output)
    print(f"saved {args.output}")
    print(json.dumps(report["quality_checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
