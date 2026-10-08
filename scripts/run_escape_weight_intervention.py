from __future__ import annotations

import argparse
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
from run_weight_intervention import (
    CONDITIONS,
    SIDES,
    batches,
    merge_blocks,
    population_counts,
    sha256_file,
    summarize,
)


def run_block(
    brain: FlyBrain,
    config: DecoderConfig,
    populations: dict[str, np.ndarray],
    stimulus: np.ndarray,
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
    action_controls = np.zeros(brain.batch, dtype=np.int32)
    first_action = np.full(brain.batch, np.nan, dtype=float)
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
            if not decoded.action:
                continue
            action_controls[batch_index] += 1
            if np.isnan(first_action[batch_index]):
                first_action[batch_index] = (step + 1) * config.dt

    seconds = stimulus_steps * config.dt
    controls = max(stimulus_steps // config.publish_steps, 1)
    triggered = np.isfinite(first_action)
    latency_or_censor = np.where(triggered, first_action, seconds + config.publish_steps * config.dt)
    average_rates = {
        name: total_counts[name] / len(populations[name]) / seconds
        for name in output_names
    }
    return {
        **{name: values.tolist() for name, values in average_rates.items()},
        "stimulus_rate_hz": (total_counts["stimulus"] / len(stimulus) / seconds).tolist(),
        "action_triggered": triggered.astype(float).tolist(),
        "action_latency_or_censor_seconds": latency_or_censor.tolist(),
        "action_control_fraction": (action_controls / controls).tolist(),
        "whole_brain_spikes": total_spikes.astype(float).tolist(),
        "whole_brain_rate_hz_per_neuron": (total_spikes / brain.n / seconds).tolist(),
        "peak_fraction_spiking_per_step": peak_fraction.tolist(),
    }


def summarize_side(raw: dict[str, list[float]]) -> dict[str, object]:
    return {key: summarize(values) for key, values in raw.items()}


def side_mean(condition: dict[str, object], metric: str) -> float:
    return float(np.mean([
        condition["sides"][side][metric]["mean"] for side in SIDES
    ]))


def ipsilateral_rate(condition: dict[str, object]) -> float:
    return float(np.mean([
        condition["sides"]["left"]["escape_left"]["mean"],
        condition["sides"]["right"]["escape_right"]["mean"],
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

    brain = FlyBrain(
        device=device,
        batch=batch,
        seed=seeds[0],
        dt=config.dt,
        sensory_input=False,
    )
    scaler = OutgoingWeightScaler(brain)
    looming = brain.cells(["LC4", "LPLC2"])
    stimuli = {
        side: brain.cells(["LC4", "LPLC2"], side=side[0].upper())
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
                "presynaptic_neurons": len(looming),
                "affected_connections": 0,
            }
        else:
            stats = scaler.apply(looming, factor)
            intervention = {"applied": True, **stats.to_dict()}

        sides = {}
        for side in SIDES:
            blocks = [
                run_block(
                    brain=brain,
                    config=config,
                    populations=populations,
                    stimulus=stimuli[side],
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

    dose_labels = ("0x", "0.5x", "baseline", "2x")
    dose_rates = [ipsilateral_rate(condition_results[label]) for label in dose_labels]
    dose_triggers = [side_mean(condition_results[label], "action_triggered") for label in dose_labels]
    dose_latencies = [
        side_mean(condition_results[label], "action_latency_or_censor_seconds")
        for label in dose_labels
    ]
    baseline_rate = ipsilateral_rate(condition_results["baseline"])
    sham_rate = ipsilateral_rate(condition_results["sham"])
    baseline_trigger = side_mean(condition_results["baseline"], "action_triggered")
    sham_trigger = side_mean(condition_results["sham"], "action_triggered")
    sham_rate_difference = abs(baseline_rate - sham_rate)
    sham_trigger_difference = abs(baseline_trigger - sham_trigger)
    sham_equivalent = sham_rate_difference <= 1.0 and sham_trigger_difference <= 0.05
    rate_monotonic = all(a <= b + 1e-9 for a, b in zip(dose_rates, dose_rates[1:]))
    trigger_monotonic = all(a <= b + 1e-9 for a, b in zip(dose_triggers, dose_triggers[1:]))
    latency_monotonic = all(a >= b - 1e-9 for a, b in zip(dose_latencies, dose_latencies[1:]))
    max_peak_fraction = max(
        condition_results["2x"]["sides"][side]["peak_fraction_spiking_per_step"]["max"]
        for side in SIDES
    )
    causal_effect = baseline_rate - dose_rates[0] >= 5.0
    baseline_reliable = baseline_trigger >= 0.95
    network_stable = bool(max_peak_fraction < 0.10)

    report = {
        "schema_version": 1,
        "experiment": "LC4/LPLC2 outgoing-weight intervention to DNp01 escape action",
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
            "decoder_sha256": sha256_file(decoder_path),
            "decoder_parameters": asdict(config),
        },
        "protocol": {
            "conditions": [label for label, _ in CONDITIONS],
            "intervention": "bilateral LC4/LPLC2 outgoing synapses, scaled in memory from one immutable base copy",
            "looming_trials": list(SIDES),
            "seeds": seeds,
            "parallel_flies_per_seed": batch,
            "trials_per_side_per_condition": len(seeds) * batch,
            "dt_seconds": config.dt,
            "warmup_steps": warmup_steps,
            "stimulus_steps": stimulus_steps,
            "stimulus_strength": strength,
            "decoder_window_steps": config.window_steps,
            "decoder_publish_steps": config.publish_steps,
            "no_action_latency_censor_seconds": stimulus_steps * config.dt + config.publish_steps * config.dt,
        },
        "population_sizes": {
            "LC4_LPLC2_all": len(looming),
            **{f"LC4_LPLC2_{side}": len(population) for side, population in stimuli.items()},
            **{name: len(population) for name, population in populations.items()},
            "whole_brain": brain.n,
        },
        "conditions": condition_results,
        "quality_checks": {
            "original_weights_unchanged": weights_checksum_before == weights_checksum_after,
            "sham_equivalence_tolerance": {
                "ipsilateral_DNp01_mean_hz": 1.0,
                "action_trigger_rate": 0.05,
            },
            "sham_ipsilateral_DNp01_difference_hz": round(sham_rate_difference, 6),
            "sham_action_trigger_difference": round(sham_trigger_difference, 6),
            "sham_equivalent_to_baseline": sham_equivalent,
            "baseline_action_trigger_rate": round(baseline_trigger, 6),
            "baseline_escape_reproducible": baseline_reliable,
            "ipsilateral_DNp01_rate_by_dose_hz": dict(zip(dose_labels, np.round(dose_rates, 6))),
            "action_trigger_rate_by_dose": dict(zip(dose_labels, np.round(dose_triggers, 6))),
            "action_latency_or_censor_by_dose_seconds": dict(zip(dose_labels, np.round(dose_latencies, 6))),
            "DNp01_rate_is_monotonic": rate_monotonic,
            "action_trigger_rate_is_monotonic": trigger_monotonic,
            "action_latency_is_monotonic": latency_monotonic,
            "zero_vs_baseline_DNp01_effect_hz": round(baseline_rate - dose_rates[0], 6),
            "causal_effect_detected": causal_effect,
            "double_weight_peak_fraction_spiking": round(float(max_peak_fraction), 8),
            "double_weight_network_not_globally_saturated": network_stable,
            "experiment_passed": bool(
                weights_checksum_before == weights_checksum_after
                and sham_equivalent
                and baseline_reliable
                and rate_monotonic
                and trigger_monotonic
                and latency_monotonic
                and causal_effect
                and network_stable
            ),
        },
        "interpretation_limits": [
            "The intervention scales all outgoing LC4 and LPLC2 synapses, not a single direct edge to DNp01.",
            "Decoder action is an escape authorization signal; vertical direction remains a non-neural world policy.",
            "No-action latency is right-censored at the protocol value stored above.",
            "A causal dose response in this LIF model does not establish biological effect size.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five-condition LC4/LPLC2 weight intervention")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seeds", default="1101,2203,3307")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "looming_weight_intervention.json",
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
