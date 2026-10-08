from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from app.config import DecoderConfig
from app.interventions import OutgoingWeightScaler
from run_reward_plasticity import evaluate_side, selected_edges
from run_weight_intervention import batches, sha256_file


CONDITIONS = ("frozen", "shuffled_pairing", "causal_stdp")
SIDES = ("left", "right")


def pair_timing_increment(
    before_masks: np.ndarray,
    now_masks: np.ndarray,
    edge_pre: np.ndarray,
    edge_post: np.ndarray,
    post_batch_shift: int = 0,
) -> np.ndarray:
    """Return one local pre/post timing update per selected sparse edge."""
    post_now = np.roll(now_masks, post_batch_shift, axis=0)
    post_before = np.roll(before_masks, post_batch_shift, axis=0)
    causal = before_masks[:, edge_pre] & post_now[:, edge_post]
    anti_causal = now_masks[:, edge_pre] & post_before[:, edge_post]
    return (causal.astype(np.float32) - 0.5 * anti_causal.astype(np.float32)).mean(axis=0)


def timing_episode(
    brain: FlyBrain,
    stimulus: np.ndarray,
    dnal: np.ndarray,
    dnar: np.ndarray,
    edge_pre: np.ndarray,
    edge_post: np.ndarray,
    expected_sign: int,
    condition: str,
    seed: int,
    strength: float,
    warmup_steps: int,
    stimulus_steps: int,
) -> tuple[np.ndarray, dict[str, float]]:
    brain.reset(seed)
    for _ in range(warmup_steps):
        brain.step()

    rng = np.random.default_rng(seed + 1701)
    eligibility = np.zeros(len(edge_pre), dtype=np.float32)
    left_counts = np.zeros(brain.batch, dtype=np.int16)
    right_counts = np.zeros(brain.batch, dtype=np.int16)
    previous = [np.empty(0, dtype=np.int64) for _ in range(brain.batch)]
    before_masks = np.zeros((brain.batch, brain.n), dtype=bool)
    now_masks = np.zeros((brain.batch, brain.n), dtype=bool)

    for _ in range(stimulus_steps):
        current = batches(brain.step(inject=((stimulus, strength),)), brain.batch)
        before_masks.fill(False)
        now_masks.fill(False)
        for batch_index, (before, now) in enumerate(zip(previous, current)):
            before_masks[batch_index, before] = True
            now_masks[batch_index, now] = True
            left_counts[batch_index] += int(np.isin(now, dnal).sum())
            right_counts[batch_index] += int(np.isin(now, dnar).sum())

        shift = 0
        if condition == "shuffled_pairing":
            shift = int(rng.integers(1, brain.batch))
        eligibility += pair_timing_increment(
            before_masks, now_masks, edge_pre, edge_post, post_batch_shift=shift
        )
        previous = current

    seconds = stimulus_steps * brain.dt
    left_hz = left_counts / len(dnal) / seconds
    right_hz = right_counts / len(dnar) / seconds
    margin = expected_sign * (right_hz - left_hz)
    return eligibility / stimulus_steps, {
        "direction_accuracy": float(np.mean(margin > 0.0)),
        "mean_margin_hz": float(margin.mean()),
    }


def run_experiment(
    device: str,
    batch: int,
    epochs: int,
    train_seed: int,
    test_seeds: list[int],
    output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
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
        side: brain.cells(["LC10a"], side=side[0].upper())
        for side in SIDES
    }
    dnal = brain.cells(["DNa02"], side="L")
    dnar = brain.cells(["DNa02"], side="R")
    required = {"LC10a": lc10a, **stimuli, "DNa02_L": dnal, "DNa02_R": dnar}
    empty = [name for name, population in required.items() if len(population) == 0]
    if empty:
        raise RuntimeError(f"Required populations not found: {', '.join(empty)}")

    edge_positions, edge_pre, edge_post = selected_edges(brain, lc10a)
    base_selected = scaler.base_weights[edge_positions].copy()
    warmup_steps = 50
    stimulus_steps = 50
    strength = 0.35
    learning_rate = 1.0
    results: dict[str, object] = {}
    learned_multipliers: dict[str, np.ndarray] = {}

    for condition in CONDITIONS:
        scaler.restore()
        multipliers = np.ones(len(edge_positions), dtype=np.float32)
        history = []
        for epoch in range(epochs):
            epoch_increment = np.zeros(len(edge_positions), dtype=np.float32)
            epoch_metrics = []
            for side_index, side in enumerate(SIDES):
                increment, metrics = timing_episode(
                    brain=brain,
                    stimulus=stimuli[side],
                    dnal=dnal,
                    dnar=dnar,
                    edge_pre=edge_pre,
                    edge_post=edge_post,
                    expected_sign=-1 if side == "left" else 1,
                    condition=condition,
                    seed=train_seed + epoch * 100 + side_index,
                    strength=strength,
                    warmup_steps=warmup_steps,
                    stimulus_steps=stimulus_steps,
                )
                epoch_increment += increment
                epoch_metrics.append(metrics)

            if condition != "frozen":
                multipliers = np.clip(
                    multipliers + learning_rate * epoch_increment, 0.5, 1.5
                )
                weights = scaler.base_weights.copy()
                weights[edge_positions] = base_selected * multipliers
                scaler.install(weights)
            history.append({
                "epoch": epoch + 1,
                "training_direction_accuracy": round(float(np.mean([
                    item["direction_accuracy"] for item in epoch_metrics
                ])), 6),
                "training_margin_hz": round(float(np.mean([
                    item["mean_margin_hz"] for item in epoch_metrics
                ])), 6),
                "mean_weight_multiplier": round(float(multipliers.mean()), 6),
                "std_weight_multiplier": round(float(multipliers.std()), 6),
            })

        learned_multipliers[condition] = multipliers.copy()
        sides = {
            side: evaluate_side(
                brain=brain,
                config=config,
                stimulus=stimuli[side],
                dnal=dnal,
                dnar=dnar,
                expected_sign=-1 if side == "left" else 1,
                seeds=test_seeds,
                strength=strength,
                warmup_steps=warmup_steps,
                stimulus_steps=stimulus_steps,
            )
            for side in SIDES
        }
        results[condition] = {
            "training_history": history,
            "weights": {
                "selected_synapses": len(edge_positions),
                "mean_multiplier": round(float(multipliers.mean()), 6),
                "std_multiplier": round(float(multipliers.std()), 6),
                "minimum_multiplier": round(float(multipliers.min()), 6),
                "maximum_multiplier": round(float(multipliers.max()), 6),
                "fraction_changed": round(float(np.mean(
                    np.abs(multipliers - 1.0) > 1e-6
                )), 6),
            },
            "sides": sides,
        }

    checksum_after = sha256_file(weights_path)
    if checksum_before != checksum_after:
        raise RuntimeError("Original weights.npz changed during local STDP experiment")

    causal = learned_multipliers["causal_stdp"]
    shuffled = learned_multipliers["shuffled_pairing"]
    rms_difference = float(np.sqrt(np.mean((causal - shuffled) ** 2)))
    peak = max(
        results["causal_stdp"]["sides"][side]["peak_fraction_spiking_per_step"]["mean"]
        for side in SIDES
    )
    frozen_accuracy = float(np.mean([
        results["frozen"]["sides"][side]["direction_accuracy"] for side in SIDES
    ]))
    causal_accuracy = float(np.mean([
        results["causal_stdp"]["sides"][side]["direction_accuracy"] for side in SIDES
    ]))
    quality = {
        "original_weights_unchanged": checksum_before == checksum_after,
        "frozen_weights_unchanged": bool(np.all(learned_multipliers["frozen"] == 1.0)),
        "causal_weights_changed": float(np.mean(np.abs(causal - 1.0) > 1e-6)) >= 0.10,
        "causal_differs_from_shuffled_pairing": rms_difference >= 0.01,
        "held_out_direction_accuracy_not_degraded": causal_accuracy >= frozen_accuracy - 0.10,
        "causal_network_not_globally_saturated": peak < 0.10,
    }
    quality["experiment_passed"] = all(quality.values())

    def mean_metric(condition: str, metric: str) -> float:
        return float(np.mean([
            results[condition]["sides"][side][metric]["mean"] for side in SIDES
        ]))

    report = {
        "schema_version": 1,
        "experiment": "Localized unsupervised pair-timing plasticity on LC10a outgoing synapses",
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
            "conditions": list(CONDITIONS),
            "epochs": epochs,
            "parallel_flies": batch,
            "train_seed_base": train_seed,
            "held_out_test_seeds": test_seeds,
            "stimulus_strength": strength,
            "warmup_seconds": warmup_steps * brain.dt,
            "stimulus_seconds": stimulus_steps * brain.dt,
            "learning_rate": learning_rate,
            "rule": "pre(t-1)*post(t) - 0.5*post(t-1)*pre(t)",
            "shuffled_control": "postsynaptic spikes circularly shifted across flies each step",
            "weight_bounds": [0.5, 1.5],
            "decoder_frozen": True,
        },
        "population_sizes": {
            "LC10a": len(lc10a),
            "selected_outgoing_synapses": len(edge_positions),
            "DNa02_left": len(dnal),
            "DNa02_right": len(dnar),
        },
        "conditions": results,
        "comparison": {
            "causal_vs_shuffled_multiplier_rms": round(rms_difference, 6),
            "mean_turn_magnitude": {
                condition: round(mean_metric(condition, "turn_magnitude"), 6)
                for condition in CONDITIONS
            },
            "mean_ipsilateral_DNa02_hz": {
                condition: round(mean_metric(condition, "ipsilateral_DNa02_hz"), 6)
                for condition in CONDITIONS
            },
            "mean_direction_accuracy": {
                "frozen": round(frozen_accuracy, 6),
                "causal_stdp": round(causal_accuracy, 6),
            },
        },
        "quality_checks": quality,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limits": [
            "This experiment tests a local computational STDP rule; it does not establish that real LC10a outputs use this rule.",
            "There are no direct LC10a-to-DNa02 edges in the installed graph, so plasticity is applied to all first-order LC10a outgoing edges.",
            "The shuffled control preserves neuron identities and per-population stimulation while breaking within-fly pre/post pairing.",
            "Behavior improvement is not a success requirement for unsupervised STDP and must not be inferred from weight change alone.",
            "Weights exist only in process memory; original connectome files are never modified.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run localized STDP on LC10a outputs")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=13101)
    parser.add_argument("--test-seeds", default="14101,14203,14307")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "local_stdp.json",
    )
    args = parser.parse_args()
    test_seeds = [int(value) for value in args.test_seeds.split(",") if value]
    if args.batch < 2 or args.epochs < 1 or not test_seeds:
        parser.error("batch must be at least 2; epochs and test seeds must be non-empty")
    report = run_experiment(
        args.device, args.batch, args.epochs, args.train_seed, test_seeds, args.output
    )
    print(f"saved {args.output}")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(json.dumps(report["quality_checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
