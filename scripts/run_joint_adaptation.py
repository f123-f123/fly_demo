from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from flybrain import FlyBrain
from flybrain.data import DATA
from flybrain.reservoir import Readout, auc

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from app.config import DecoderConfig
from app.decoder import RuleDecoder
from app.interventions import OutgoingWeightScaler
from run_local_stdp import SIDES, selected_edges, timing_episode
from run_weight_intervention import batches, sha256_file


def collect_steering_records(
    brain: FlyBrain,
    seeds: list[int],
    strength: float,
    warmup_steps: int,
    stimulus_steps: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    stimuli = {
        side: brain.cells(["LC10a"], side=side[0].upper()) for side in SIDES
    }
    dnal = brain.cells(["DNa02"], side="L")
    dnar = brain.cells(["DNa02"], side="R")
    descending = brain.cells(["descending_neuron"])
    required = {**stimuli, "DNa02_L": dnal, "DNa02_R": dnar, "descending": descending}
    empty = [name for name, population in required.items() if len(population) == 0]
    if empty:
        raise RuntimeError(f"Required populations not found: {', '.join(empty)}")

    slot = np.full(brain.n, -1, dtype=np.int32)
    slot[descending] = np.arange(len(descending), dtype=np.int32)
    trace_decay = np.float32(np.exp(-brain.dt / 0.1))
    records: list[dict[str, object]] = []
    for side in SIDES:
        for seed in seeds:
            brain.reset(seed)
            for _ in range(warmup_steps):
                brain.step()
            trace = np.zeros((len(descending), brain.batch), dtype=np.float32)
            left_counts = np.zeros(brain.batch, dtype=np.int16)
            right_counts = np.zeros(brain.batch, dtype=np.int16)
            total_counts = np.zeros(brain.batch, dtype=np.int64)
            peaks = np.zeros(brain.batch, dtype=float)
            for _ in range(stimulus_steps):
                fired_by_fly = batches(
                    brain.step(inject=((stimuli[side], strength),)), brain.batch
                )
                trace *= trace_decay
                for batch_index, fired in enumerate(fired_by_fly):
                    slots = slot[fired]
                    slots = slots[slots >= 0]
                    trace[slots, batch_index] += 1.0
                    left_counts[batch_index] += int(np.isin(fired, dnal).sum())
                    right_counts[batch_index] += int(np.isin(fired, dnar).sum())
                    total_counts[batch_index] += len(fired)
                    peaks[batch_index] = max(peaks[batch_index], len(fired) / brain.n)
            seconds = stimulus_steps * brain.dt
            for batch_index in range(brain.batch):
                records.append({
                    "side": side,
                    "label": int(side == "right"),
                    "seed": seed,
                    "parallel_index": batch_index,
                    "descending_trace": trace[:, batch_index].copy(),
                    "steer_left_hz": float(left_counts[batch_index] / len(dnal) / seconds),
                    "steer_right_hz": float(right_counts[batch_index] / len(dnar) / seconds),
                    "whole_brain_rate_hz_per_neuron": float(
                        total_counts[batch_index] / brain.n / seconds
                    ),
                    "peak_fraction_spiking_per_step": float(peaks[batch_index]),
                })
    return records, {"left": dnal, "right": dnar, "descending": descending}


def fit_readout(records: list[dict[str, object]]) -> Readout:
    return Readout.fit(
        np.stack([record["descending_trace"] for record in records]),
        np.asarray([record["label"] for record in records]),
        kind="logistic",
        groups=np.asarray([record["seed"] for record in records]),
        components=(5, 20, 40),
        lambdas=(0.01, 0.1, 1.0, 10.0),
    )


def evaluate_readout(readout: Readout, records: list[dict[str, object]]) -> dict[str, float]:
    features = np.stack([record["descending_trace"] for record in records])
    labels = np.asarray([record["label"] for record in records])
    probability = np.asarray(readout.predict(features))
    predicted = probability >= 0.5
    return {
        "samples": len(records),
        "auc": round(float(auc(labels, probability)), 6),
        "accuracy": round(float(np.mean(predicted == labels)), 6),
        "true_positive_rate": round(float(np.mean(predicted[labels == 1])), 6),
        "false_positive_rate": round(float(np.mean(predicted[labels == 0])), 6),
    }


def evaluate_rule(
    config: DecoderConfig, records: list[dict[str, object]]
) -> dict[str, float]:
    correct = []
    turns = []
    for record in records:
        decoder = RuleDecoder(copy.deepcopy(config))
        decoded = decoder.decode({
            "steer_left": record["steer_left_hz"],
            "steer_right": record["steer_right_hz"],
            "escape_left": 0.0,
            "escape_right": 0.0,
        }, config.publish_steps * config.dt)
        expected_sign = -1 if record["side"] == "left" else 1
        correct.append(float(decoded.raw_turn * expected_sign > 0.0))
        turns.append(abs(decoded.raw_turn))
    return {
        "samples": len(records),
        "direction_accuracy": round(float(np.mean(correct)), 6),
        "mean_turn_magnitude": round(float(np.mean(turns)), 6),
    }


def network_summary(records: list[dict[str, object]]) -> dict[str, float]:
    rates = np.asarray([record["whole_brain_rate_hz_per_neuron"] for record in records])
    peaks = np.asarray([record["peak_fraction_spiking_per_step"] for record in records])
    return {
        "whole_brain_rate_hz_per_neuron": round(float(rates.mean()), 6),
        "peak_fraction_spiking_per_step": round(float(peaks.max()), 6),
    }


def run_experiment(
    device: str,
    batch: int,
    epochs: int,
    train_seed: int,
    train_seeds: list[int],
    test_seeds: list[int],
    output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    if set(train_seeds) & set(test_seeds):
        raise ValueError("Training and held-out test seeds must not overlap")
    config = DecoderConfig.load()
    weights_path = Path(DATA) / "weights.npz"
    checksum_before = sha256_file(weights_path)
    brain = FlyBrain(
        device=device,
        batch=batch,
        seed=train_seed,
        dt=config.dt,
        sensory_input=False,
    )
    scaler = OutgoingWeightScaler(brain)
    lc10a = brain.cells(["LC10a"])
    stimuli = {
        side: brain.cells(["LC10a"], side=side[0].upper()) for side in SIDES
    }
    dnal = brain.cells(["DNa02"], side="L")
    dnar = brain.cells(["DNa02"], side="R")
    edge_positions, edge_pre, edge_post = selected_edges(brain, lc10a)
    base_selected = scaler.base_weights[edge_positions].copy()
    warmup_steps = 50
    stimulus_steps = 50
    strength = 0.35
    learning_rate = 1.0

    frozen_train, populations = collect_steering_records(
        brain, train_seeds, strength, warmup_steps, stimulus_steps
    )
    frozen_test, _ = collect_steering_records(
        brain, test_seeds, strength, warmup_steps, stimulus_steps
    )
    frozen_readout = fit_readout(frozen_train)

    multipliers = np.ones(len(edge_positions), dtype=np.float32)
    training_history = []
    for epoch in range(epochs):
        epoch_increment = np.zeros(len(edge_positions), dtype=np.float32)
        metrics = []
        for side_index, side in enumerate(SIDES):
            increment, item = timing_episode(
                brain=brain,
                stimulus=stimuli[side],
                dnal=dnal,
                dnar=dnar,
                edge_pre=edge_pre,
                edge_post=edge_post,
                expected_sign=-1 if side == "left" else 1,
                condition="causal_stdp",
                seed=train_seed + epoch * 100 + side_index,
                strength=strength,
                warmup_steps=warmup_steps,
                stimulus_steps=stimulus_steps,
            )
            epoch_increment += increment
            metrics.append(item)
        multipliers = np.clip(
            multipliers + learning_rate * epoch_increment, 0.5, 1.5
        )
        weights = scaler.base_weights.copy()
        weights[edge_positions] = base_selected * multipliers
        scaler.install(weights)
        training_history.append({
            "epoch": epoch + 1,
            "training_direction_accuracy": round(float(np.mean([
                item["direction_accuracy"] for item in metrics
            ])), 6),
            "mean_weight_multiplier": round(float(multipliers.mean()), 6),
            "std_weight_multiplier": round(float(multipliers.std()), 6),
        })

    adapted_train, _ = collect_steering_records(
        brain, train_seeds, strength, warmup_steps, stimulus_steps
    )
    adapted_test, _ = collect_steering_records(
        brain, test_seeds, strength, warmup_steps, stimulus_steps
    )
    adapted_readout = fit_readout(adapted_train)

    arms = {
        "frozen_brain_rule_decoder": evaluate_rule(config, frozen_test),
        "adapted_brain_rule_decoder": evaluate_rule(config, adapted_test),
        "frozen_brain_trained_decoder": evaluate_readout(frozen_readout, frozen_test),
        "adapted_brain_frozen_decoder": evaluate_readout(frozen_readout, adapted_test),
        "adapted_brain_retrained_decoder": evaluate_readout(adapted_readout, adapted_test),
    }
    checksum_after = sha256_file(weights_path)
    if checksum_before != checksum_after:
        raise RuntimeError("Original weights.npz changed during joint-adaptation experiment")

    frozen_accuracy = arms["frozen_brain_trained_decoder"]["accuracy"]
    adapted_fixed_accuracy = arms["adapted_brain_frozen_decoder"]["accuracy"]
    joint_accuracy = arms["adapted_brain_retrained_decoder"]["accuracy"]
    best_single = max(frozen_accuracy, adapted_fixed_accuracy)
    hypothesis = {
        "joint_accuracy_gain_over_best_single_arm": round(
            float(joint_accuracy - best_single), 6
        ),
        "joint_advantage_supported": joint_accuracy - best_single >= 0.02,
    }
    adapted_network = network_summary(adapted_test)
    quality = {
        "original_weights_unchanged": checksum_before == checksum_after,
        "train_and_test_seeds_disjoint": not bool(set(train_seeds) & set(test_seeds)),
        "selected_weights_changed": float(np.mean(np.abs(multipliers - 1.0) > 1e-6)) >= 0.10,
        "frozen_brain_readout_held_out_accuracy_at_least_80_percent": frozen_accuracy >= 0.80,
        "adapted_brain_frozen_readout_accuracy_at_least_80_percent": adapted_fixed_accuracy >= 0.80,
        "joint_readout_held_out_accuracy_at_least_80_percent": joint_accuracy >= 0.80,
        "adapted_network_not_globally_saturated": adapted_network["peak_fraction_spiking_per_step"] < 0.10,
        "joint_not_materially_worse_than_best_single_arm": joint_accuracy >= best_single - 0.05,
    }
    quality["experiment_passed"] = all(quality.values())

    report = {
        "schema_version": 1,
        "experiment": "Factorial comparison of fixed-brain decoder training, local STDP, and joint adaptation",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provenance": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "flybrain_version": importlib.metadata.version("flybrain"),
            "device": brain.device,
            "weights_sha256_before": checksum_before,
            "weights_sha256_after": checksum_after,
            "decoder_sha256": sha256_file(ROOT / "config" / "decoder.json"),
            "decoder_parameters": asdict(config),
        },
        "protocol": {
            "parallel_flies": batch,
            "stdp_epochs": epochs,
            "stdp_train_seed_base": train_seed,
            "readout_train_seeds": train_seeds,
            "held_out_test_seeds": test_seeds,
            "stimulus_strength": strength,
            "warmup_seconds": warmup_steps * brain.dt,
            "stimulus_seconds": stimulus_steps * brain.dt,
            "stdp_rule": "pre(t-1)*post(t) - 0.5*post(t-1)*pre(t)",
            "stdp_weight_bounds": [0.5, 1.5],
            "readout": "PCA plus L2 logistic regression with seed-grouped cross-validation",
            "reward_controls_source": "config/experiments/reward_plasticity.json",
        },
        "population_sizes": {
            "LC10a": len(lc10a),
            "selected_outgoing_synapses": len(edge_positions),
            "descending_features": len(populations["descending"]),
        },
        "stdp_weights": {
            "mean_multiplier": round(float(multipliers.mean()), 6),
            "std_multiplier": round(float(multipliers.std()), 6),
            "minimum_multiplier": round(float(multipliers.min()), 6),
            "maximum_multiplier": round(float(multipliers.max()), 6),
            "fraction_changed": round(float(np.mean(np.abs(multipliers - 1.0) > 1e-6)), 6),
            "training_history": training_history,
        },
        "readout_training": {
            "frozen_brain": {
                "components": frozen_readout.components,
                "lambda": frozen_readout.lam,
                "cross_validated_auc": round(float(frozen_readout.cv_score), 6),
            },
            "adapted_brain": {
                "components": adapted_readout.components,
                "lambda": adapted_readout.lam,
                "cross_validated_auc": round(float(adapted_readout.cv_score), 6),
            },
        },
        "arms": arms,
        "network": {
            "frozen_test": network_summary(frozen_test),
            "adapted_test": adapted_network,
        },
        "hypothesis_test": hypothesis,
        "quality_checks": quality,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limits": [
            "Joint adaptation here means local unsupervised LC10a-output STDP followed by refitting a linear descending-neuron readout.",
            "It is not simultaneous online co-learning and does not use a biological reward circuit.",
            "The separately recorded reward experiment supplies frozen, sham-reward, and random-reward controls and failed its learning-effect gate.",
            "A passed execution-quality gate does not imply that joint adaptation outperforms either single-adaptation arm; that is reported separately.",
            "All connectome weight changes exist only in process memory.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed/adapted brain and decoder factorial experiment")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=15101)
    parser.add_argument("--readout-train-seeds", default="16101,16203,16307")
    parser.add_argument("--test-seeds", default="17101,17203,17307")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "joint_adaptation.json",
    )
    args = parser.parse_args()
    train_seeds = [int(value) for value in args.readout_train_seeds.split(",") if value]
    test_seeds = [int(value) for value in args.test_seeds.split(",") if value]
    if args.batch < 1 or args.epochs < 1 or not train_seeds or not test_seeds:
        parser.error("batch, epochs and seed sets must be non-empty")
    report = run_experiment(
        args.device,
        args.batch,
        args.epochs,
        args.train_seed,
        train_seeds,
        test_seeds,
        args.output,
    )
    print(f"saved {args.output}")
    print(json.dumps(report["arms"], ensure_ascii=False, indent=2))
    print(json.dumps(report["hypothesis_test"], ensure_ascii=False, indent=2))
    print(json.dumps(report["quality_checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
