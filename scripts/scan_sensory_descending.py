from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from flybrain import FlyBrain
from flybrain.data import DATA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_multisensory_encoding import CHANNELS
from run_weight_intervention import batches, sha256_file


def collect_rates(
    brain: FlyBrain,
    descending: np.ndarray,
    slot: np.ndarray,
    seeds: list[int],
    stimulus: np.ndarray | None,
    strength: float,
    warmup_steps: int,
    stimulus_steps: int,
) -> np.ndarray:
    samples = []
    for seed in seeds:
        brain.reset(seed)
        for _ in range(warmup_steps):
            brain.step()
        counts = np.zeros((len(descending), brain.batch), dtype=np.int16)
        injection = () if stimulus is None else ((stimulus, strength),)
        for _ in range(stimulus_steps):
            fired_by_fly = batches(brain.step(inject=injection), brain.batch)
            for batch_index, fired in enumerate(fired_by_fly):
                positions = slot[fired]
                positions = positions[positions >= 0]
                counts[positions, batch_index] += 1
        rates = counts.T / (stimulus_steps * brain.dt)
        samples.append(rates)
    return np.concatenate(samples, axis=0)


def candidate_rows(
    brain: FlyBrain,
    descending: np.ndarray,
    baseline: np.ndarray,
    stimulated: np.ndarray,
    seed_count: int,
    batch: int,
) -> list[dict[str, object]]:
    baseline_mean = baseline.mean(axis=0)
    stimulated_mean = stimulated.mean(axis=0)
    delta = stimulated_mean - baseline_mean
    pooled = np.sqrt((baseline.var(axis=0) + stimulated.var(axis=0)) / 2.0)
    effect = delta / np.maximum(pooled, 0.25)
    baseline_threshold = baseline_mean + 2.0 * baseline.std(axis=0)
    response_fraction = (stimulated > baseline_threshold).mean(axis=0)
    seed_means = stimulated.reshape(seed_count, batch, -1).mean(axis=1)
    baseline_seed_means = baseline.reshape(seed_count, batch, -1).mean(axis=1)
    positive_seed_fraction = (seed_means > baseline_seed_means).mean(axis=0)

    rows = []
    for local_index, neuron in enumerate(descending):
        rows.append({
            "neuron_index": int(neuron),
            "cell_type": str(brain.cell_type[neuron]),
            "side": str(brain.side[neuron]),
            "baseline_hz": round(float(baseline_mean[local_index]), 6),
            "stimulated_hz": round(float(stimulated_mean[local_index]), 6),
            "delta_hz": round(float(delta[local_index]), 6),
            "effect_size": round(float(effect[local_index]), 6),
            "response_fraction": round(float(response_fraction[local_index]), 6),
            "positive_seed_fraction": round(float(positive_seed_fraction[local_index]), 6),
        })
    rows.sort(key=lambda row: (row["delta_hz"], row["effect_size"]), reverse=True)
    return rows


def run_experiment(
    device: str,
    batch: int,
    seeds: list[int],
    output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    weights_path = Path(DATA) / "weights.npz"
    checksum_before = sha256_file(weights_path)
    brain = FlyBrain(
        device=device,
        batch=batch,
        seed=seeds[0],
        dt=0.02,
        sensory_input=False,
    )
    descending = brain.cells(["descending_neuron"])
    slot = np.full(brain.n, -1, dtype=np.int32)
    slot[descending] = np.arange(len(descending), dtype=np.int32)
    inputs = {
        channel: brain.cells(spec["input"])
        for channel, spec in CHANNELS.items()
    }
    empty = [name for name, population in {"descending": descending, **inputs}.items() if len(population) == 0]
    if empty:
        raise RuntimeError(f"Required populations not found: {', '.join(empty)}")

    warmup_steps = 50
    stimulus_steps = 50
    baseline = collect_rates(
        brain, descending, slot, seeds, None, 0.0, warmup_steps, stimulus_steps
    )
    all_rows: dict[str, list[dict[str, object]]] = {}
    stimulated_rates: dict[str, np.ndarray] = {}
    for channel, stimulus in inputs.items():
        rates = collect_rates(
            brain, descending, slot, seeds, stimulus, 0.8, warmup_steps, stimulus_steps
        )
        stimulated_rates[channel] = rates
        all_rows[channel] = candidate_rows(
            brain, descending, baseline, rates, len(seeds), batch
        )

    delta_by_channel = {
        channel: rates.mean(axis=0) - baseline.mean(axis=0)
        for channel, rates in stimulated_rates.items()
    }
    channels = {}
    for channel, rows in all_rows.items():
        for row in rows:
            local_index = int(np.flatnonzero(descending == row["neuron_index"])[0])
            other_deltas = [
                max(0.0, float(delta[local_index]))
                for other, delta in delta_by_channel.items()
                if other != channel
            ]
            row["channel_specificity_ratio"] = round(
                max(0.0, row["delta_hz"]) / max(max(other_deltas, default=0.0), 0.25),
                6,
            )
        stable = [
            row for row in rows
            if row["delta_hz"] >= 1.0
            and row["effect_size"] >= 2.0
            and row["positive_seed_fraction"] == 1.0
        ]
        channels[channel] = {
            "input_types": CHANNELS[channel]["input"],
            "input_population_size": len(inputs[channel]),
            "stable_candidate_count": len(stable),
            "top_20_by_delta": rows[:20],
            "top_20_stable": stable[:20],
        }

    checksum_after = sha256_file(weights_path)
    if checksum_before != checksum_after:
        raise RuntimeError("Original weights.npz changed during descending scan")
    quality = {
        "original_weights_unchanged": checksum_before == checksum_after,
        "all_channels_scanned": len(channels) == len(CHANNELS),
        "all_channels_have_stable_candidates": all(
            result["stable_candidate_count"] > 0 for result in channels.values()
        ),
    }
    quality["scan_passed"] = all(quality.values())
    report = {
        "schema_version": 1,
        "experiment": "Whole-descending-neuron scan after software sensory stimulation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provenance": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "flybrain_version": importlib.metadata.version("flybrain"),
            "device": brain.device,
            "weights_sha256_before": checksum_before,
            "weights_sha256_after": checksum_after,
        },
        "protocol": {
            "seeds": seeds,
            "parallel_flies_per_seed": batch,
            "samples_per_condition": len(seeds) * batch,
            "warmup_seconds": warmup_steps * brain.dt,
            "stimulus_seconds": stimulus_steps * brain.dt,
            "stimulus_strength": 0.8,
            "encoder": "constant intensity",
            "stable_candidate_rule": "delta >= 1 Hz, effect size >= 2, positive mean delta in every seed",
        },
        "descending_population_size": len(descending),
        "channels": channels,
        "quality_checks": quality,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limits": [
            "Candidate means statistically stable response in this simulation, not a validated behavioral role.",
            "Cell-type role must be verified from primary biological sources before any candidate controls movement.",
            "The scan ranks individual neurons; repeated cell types can appear more than once.",
            "No candidate is connected to the Decoder by this script.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan descending-neuron responses to new sensory channels")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seeds", default="8101,8203,8307")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "sensory_descending_scan.json",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if args.batch < 1 or not seeds:
        parser.error("batch and seeds must be non-empty")
    report = run_experiment(args.device, args.batch, seeds, args.output)
    print(f"saved {args.output}")
    print(json.dumps(report["quality_checks"], ensure_ascii=False, indent=2))
    for channel, result in report["channels"].items():
        names = [row["cell_type"] for row in result["top_20_stable"][:5]]
        print(channel, result["stable_candidate_count"], names)


if __name__ == "__main__":
    main()
