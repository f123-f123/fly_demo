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
from flybrain.reservoir import Readout, auc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.interventions import OutgoingWeightScaler
from run_multisensory_encoding import CHANNELS
from run_weight_intervention import batches, sha256_file


SENSORY_CHANNELS = ("warming", "cooling", "drying", "moistening")
CONDITIONS = (("baseline", None), ("sham", 1.0), ("0x", 0.0))


def collect_condition(
    brain: FlyBrain,
    descending: np.ndarray,
    slot: np.ndarray,
    inputs: dict[str, np.ndarray],
    dnb05: np.ndarray,
    seeds: list[int],
    warmup_steps: int,
    stimulus_steps: int,
) -> dict[str, dict[str, object]]:
    result = {}
    for scenario in ("null", *SENSORY_CHANNELS):
        samples = []
        dnb05_rates = []
        injection = () if scenario == "null" else ((inputs[scenario], 0.8),)
        for seed in seeds:
            brain.reset(seed)
            for _ in range(warmup_steps):
                brain.step()
            counts = np.zeros((len(descending), brain.batch), dtype=np.int16)
            dnb_counts = np.zeros(brain.batch, dtype=np.int16)
            for _ in range(stimulus_steps):
                fired_by_fly = batches(brain.step(inject=injection), brain.batch)
                for batch_index, fired in enumerate(fired_by_fly):
                    positions = slot[fired]
                    positions = positions[positions >= 0]
                    counts[positions, batch_index] += 1
                    dnb_counts[batch_index] += int(np.isin(fired, dnb05).sum())
            samples.append((counts.T / (stimulus_steps * brain.dt)).astype(np.float32))
            dnb05_rates.append(dnb_counts / len(dnb05) / (stimulus_steps * brain.dt))
        result[scenario] = {
            "rates": np.concatenate(samples, axis=0),
            "dnb05_hz": np.concatenate(dnb05_rates),
        }
    return result


def train_readouts(
    training: dict[str, dict[str, object]],
    feature_mask: np.ndarray,
    seeds: list[int],
    batch: int,
) -> dict[str, dict[str, object]]:
    readouts = {}
    for channel in SENSORY_CHANNELS:
        null_rates = training["null"]["rates"]
        stimulated_rates = training[channel]["rates"]
        pooled = np.sqrt((null_rates.var(axis=0) + stimulated_rates.var(axis=0)) / 2.0)
        effect = np.abs(stimulated_rates.mean(axis=0) - null_rates.mean(axis=0)) / np.maximum(pooled, 0.25)
        candidates = np.flatnonzero(feature_mask)
        selected = candidates[np.argsort(effect[candidates])[-min(50, len(candidates)):]]
        X = np.concatenate([
            null_rates[:, selected],
            stimulated_rates[:, selected],
        ])
        y = np.concatenate([
            np.zeros(len(training["null"]["rates"]), dtype=np.int8),
            np.ones(len(training[channel]["rates"]), dtype=np.int8),
        ])
        groups = np.concatenate([
            np.repeat(seeds, batch),
            np.repeat(seeds, batch),
        ])
        readouts[channel] = {
            "selected": selected,
            "readout": Readout.fit(
            X,
            y,
            kind="logistic",
            groups=groups,
            components=(5, 20, 40),
            lambdas=(0.01, 0.1, 1.0, 10.0),
            ),
        }
    return readouts


def evaluate(
    condition: dict[str, dict[str, object]],
    readouts: dict[str, dict[str, object]],
) -> dict[str, object]:
    results = {}
    for channel, trained in readouts.items():
        readout = trained["readout"]
        selected = trained["selected"]
        X = np.concatenate([
            condition["null"]["rates"][:, selected],
            condition[channel]["rates"][:, selected],
        ])
        y = np.concatenate([
            np.zeros(len(condition["null"]["rates"]), dtype=np.int8),
            np.ones(len(condition[channel]["rates"]), dtype=np.int8),
        ])
        probability = np.asarray(readout.predict(X))
        predicted = probability >= 0.5
        positive = y == 1
        negative = ~positive
        dnb = np.asarray(condition[channel]["dnb05_hz"])
        dnb_null = np.asarray(condition["null"]["dnb05_hz"])
        results[channel] = {
            "held_out_auc": round(float(auc(y, probability)), 6),
            "held_out_accuracy": round(float(np.mean(predicted == y)), 6),
            "true_positive_rate": round(float(np.mean(predicted[positive])), 6),
            "false_positive_rate": round(float(np.mean(predicted[negative])), 6),
            "DNb05_stimulated_hz": round(float(dnb.mean()), 6),
            "DNb05_null_hz": round(float(dnb_null.mean()), 6),
            "DNb05_delta_hz": round(float(dnb.mean() - dnb_null.mean()), 6),
        }
    return results


def run_experiment(
    device: str,
    batch: int,
    train_seeds: list[int],
    test_seeds: list[int],
    output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    weights_path = Path(DATA) / "weights.npz"
    checksum_before = sha256_file(weights_path)
    brain = FlyBrain(
        device=device,
        batch=batch,
        seed=train_seeds[0],
        dt=0.02,
        sensory_input=False,
    )
    scaler = OutgoingWeightScaler(brain)
    descending = brain.cells(["descending_neuron"])
    dnb05 = brain.cells(["DNb05"])
    inputs = {
        channel: brain.cells(CHANNELS[channel]["input"])
        for channel in SENSORY_CHANNELS
    }
    empty = [name for name, population in {"descending": descending, "DNb05": dnb05, **inputs}.items() if len(population) == 0]
    if empty:
        raise RuntimeError(f"Required populations not found: {', '.join(empty)}")
    slot = np.full(brain.n, -1, dtype=np.int32)
    slot[descending] = np.arange(len(descending), dtype=np.int32)
    feature_mask = ~np.isin(descending, dnb05)
    warmup_steps = 50
    stimulus_steps = 50

    scaler.restore()
    training = collect_condition(
        brain,
        descending,
        slot,
        inputs,
        dnb05,
        train_seeds,
        warmup_steps,
        stimulus_steps,
    )
    readouts = train_readouts(training, feature_mask, train_seeds, batch)

    conditions = {}
    for label, factor in CONDITIONS:
        if factor is None:
            scaler.restore()
            intervention = {
                "applied": False,
                "factor": 1.0,
                "presynaptic_neurons": len(dnb05),
                "affected_connections": 0,
            }
        else:
            stats = scaler.apply(dnb05, factor)
            intervention = {"applied": True, **stats.to_dict()}
        collected = collect_condition(
            brain,
            descending,
            slot,
            inputs,
            dnb05,
            test_seeds,
            warmup_steps,
            stimulus_steps,
        )
        conditions[label] = {
            "intervention": intervention,
            "channels": evaluate(collected, readouts),
        }

    checksum_after = sha256_file(weights_path)
    if checksum_before != checksum_after:
        raise RuntimeError("Original weights.npz changed during DNb05 ablation")

    auc_drops = {
        channel: round(
            conditions["baseline"]["channels"][channel]["held_out_auc"]
            - conditions["0x"]["channels"][channel]["held_out_auc"],
            6,
        )
        for channel in SENSORY_CHANNELS
    }
    accuracy_drops = {
        channel: round(
            conditions["baseline"]["channels"][channel]["held_out_accuracy"]
            - conditions["0x"]["channels"][channel]["held_out_accuracy"],
            6,
        )
        for channel in SENSORY_CHANNELS
    }
    sham_differences = {
        channel: round(abs(
            conditions["baseline"]["channels"][channel]["held_out_auc"]
            - conditions["sham"]["channels"][channel]["held_out_auc"]
        ), 6)
        for channel in SENSORY_CHANNELS
    }
    bottleneck_supported = {
        channel: bool(auc_drops[channel] >= 0.20 or accuracy_drops[channel] >= 0.20)
        for channel in SENSORY_CHANNELS
    }
    quality = {
        "original_weights_unchanged": checksum_before == checksum_after,
        "sham_equivalent_to_baseline": all(value <= 0.05 for value in sham_differences.values()),
        "DNb05_excluded_from_readout_features": bool(feature_mask.sum() == len(descending) - len(dnb05)),
        "all_channels_have_baseline_auc_at_least_0.8": all(
            conditions["baseline"]["channels"][channel]["held_out_auc"] >= 0.80
            for channel in SENSORY_CHANNELS
        ),
        "targeted_cooling_and_moistening_have_baseline_auc_at_least_0.8": all(
            conditions["baseline"]["channels"][channel]["held_out_auc"] >= 0.80
            for channel in ("cooling", "moistening")
        ),
    }
    quality["experiment_passed"] = bool(
        quality["original_weights_unchanged"]
        and quality["sham_equivalent_to_baseline"]
        and quality["DNb05_excluded_from_readout_features"]
        and quality["targeted_cooling_and_moistening_have_baseline_auc_at_least_0.8"]
    )
    report = {
        "schema_version": 1,
        "experiment": "Causal DNb05 outgoing ablation during thermo-hygrosensory stimulation",
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
            "conditions": [label for label, _ in CONDITIONS],
            "channels": list(SENSORY_CHANNELS),
            "train_seeds": train_seeds,
            "held_out_test_seeds": test_seeds,
            "parallel_flies_per_seed": batch,
            "samples_per_channel_train": len(train_seeds) * batch,
            "samples_per_channel_test": len(test_seeds) * batch,
            "stimulus_strength": 0.8,
            "stimulus_seconds": stimulus_steps * brain.dt,
            "intervention": "bilateral DNb05 outgoing weights scaled in memory",
            "readout": "baseline-trained logistic readout over the 50 strongest training-only descending features, excluding DNb05",
            "bottleneck_support_rule": "held-out AUC or accuracy drop >= 0.20 under 0x",
        },
        "population_sizes": {
            "DNb05": len(dnb05),
            "descending_total": len(descending),
            "readout_features": int(feature_mask.sum()),
        },
        "readout_training": {
            channel: {
                "selected_features": len(trained["selected"]),
                "components": trained["readout"].components,
                "lambda": trained["readout"].lam,
                "cross_validated_auc": round(float(trained["readout"].cv_score), 6),
            }
            for channel, trained in readouts.items()
        },
        "conditions": conditions,
        "causal_comparison": {
            "baseline_minus_0x_auc": auc_drops,
            "baseline_minus_0x_accuracy": accuracy_drops,
            "baseline_vs_sham_auc_absolute_difference": sham_differences,
            "DNb05_bottleneck_supported": bottleneck_supported,
        },
        "quality_checks": quality,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limits": [
            "The intervention removes DNb05 outgoing influence but does not prevent DNb05 itself from firing.",
            "Readout features exclude DNb05, so any effect measures its causal influence on the rest of the descending population.",
            "A negative bottleneck result does not prove DNb05 has no motor role outside this LIF model or stimulus protocol.",
            "No condition controls movement.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DNb05 necessity for thermo-hygrosensory descending representations")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--train-seeds", default="9101,9203,9307")
    parser.add_argument("--test-seeds", default="10101,10203")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "dnb05_ablation.json",
    )
    args = parser.parse_args()
    train_seeds = [int(value) for value in args.train_seeds.split(",") if value]
    test_seeds = [int(value) for value in args.test_seeds.split(",") if value]
    if args.batch < 1 or not train_seeds or not test_seeds:
        parser.error("batch and both seed lists must be non-empty")
    report = run_experiment(args.device, args.batch, train_seeds, test_seeds, args.output)
    print(f"saved {args.output}")
    print(json.dumps(report["quality_checks"], ensure_ascii=False, indent=2))
    print(json.dumps(report["causal_comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
