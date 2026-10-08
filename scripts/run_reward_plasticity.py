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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DecoderConfig
from app.decoder import RuleDecoder
from app.interventions import OutgoingWeightScaler
from run_weight_intervention import batches, sha256_file


CONDITIONS = ("frozen", "sham_reward", "random_reward", "contingent_reward")
SIDES = ("left", "right")


def selected_edges(brain: FlyBrain, presynaptic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    edge_pre = []
    for neuron in np.asarray(presynaptic, dtype=np.int64):
        start = int(brain.indptr[neuron])
        stop = int(brain.indptr[neuron + 1])
        positions.append(np.arange(start, stop, dtype=np.int64))
        edge_pre.append(np.full(stop - start, neuron, dtype=np.int64))
    edge_positions = np.concatenate(positions)
    return edge_positions, np.concatenate(edge_pre), np.asarray(brain.indices)[edge_positions]


def training_episode(
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
    eligibility = np.zeros((len(edge_pre), brain.batch), dtype=np.float32)
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
        causal = before_masks[:, edge_pre] & now_masks[:, edge_post]
        anti_causal = now_masks[:, edge_pre] & before_masks[:, edge_post]
        eligibility += causal.T.astype(np.float32)
        eligibility -= 0.5 * anti_causal.T.astype(np.float32)
        previous = current

    seconds = stimulus_steps * brain.dt
    left_hz = left_counts / len(dnal) / seconds
    right_hz = right_counts / len(dnar) / seconds
    margin = expected_sign * (right_hz - left_hz)
    if condition in {"frozen", "sham_reward"}:
        rewards = np.zeros(brain.batch, dtype=np.float32)
    elif condition == "random_reward":
        rewards = np.ones(brain.batch, dtype=np.float32)
        rewards[:brain.batch // 2] = -1.0
        np.random.default_rng(seed + 404).shuffle(rewards)
    elif condition == "contingent_reward":
        rewards = np.where(margin > 0.0, 1.0, -1.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    rewarded_eligibility = (eligibility * rewards[None, :]).mean(axis=1) / stimulus_steps
    return rewarded_eligibility, {
        "mean_reward": float(rewards.mean()),
        "direction_accuracy": float(np.mean(margin > 0.0)),
        "mean_margin_hz": float(margin.mean()),
    }


def evaluate_side(
    brain: FlyBrain,
    config: DecoderConfig,
    stimulus: np.ndarray,
    dnal: np.ndarray,
    dnar: np.ndarray,
    expected_sign: int,
    seeds: list[int],
    strength: float,
    warmup_steps: int,
    stimulus_steps: int,
) -> dict[str, object]:
    left_rates = []
    right_rates = []
    turns = []
    correct = []
    network_rates = []
    peak_fractions = []
    for seed in seeds:
        brain.reset(seed)
        for _ in range(warmup_steps):
            brain.step()
        left_counts = np.zeros(brain.batch, dtype=np.int16)
        right_counts = np.zeros(brain.batch, dtype=np.int16)
        total_spikes = np.zeros(brain.batch, dtype=np.int64)
        peaks = np.zeros(brain.batch, dtype=float)
        for _ in range(stimulus_steps):
            fired_by_fly = batches(
                brain.step(inject=((stimulus, strength),)), brain.batch
            )
            for batch_index, fired in enumerate(fired_by_fly):
                left_counts[batch_index] += int(np.isin(fired, dnal).sum())
                right_counts[batch_index] += int(np.isin(fired, dnar).sum())
                total_spikes[batch_index] += len(fired)
                peaks[batch_index] = max(peaks[batch_index], len(fired) / brain.n)
        seconds = stimulus_steps * brain.dt
        left = left_counts / len(dnal) / seconds
        right = right_counts / len(dnar) / seconds
        for batch_index in range(brain.batch):
            decoder = RuleDecoder(copy.deepcopy(config))
            decoded = decoder.decode({
                "steer_left": float(left[batch_index]),
                "steer_right": float(right[batch_index]),
                "escape_left": 0.0,
                "escape_right": 0.0,
            }, config.publish_steps * config.dt)
            turns.append(decoded.raw_turn)
            correct.append(float(decoded.raw_turn * expected_sign > 0.0))
        left_rates.extend(left.tolist())
        right_rates.extend(right.tolist())
        network_rates.extend((total_spikes / brain.n / seconds).tolist())
        peak_fractions.extend(peaks.tolist())

    def summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "mean": round(float(array.mean()), 6),
            "std": round(float(array.std()), 6),
        }

    ipsilateral = left_rates if expected_sign < 0 else right_rates
    return {
        "steer_left_hz": summary(left_rates),
        "steer_right_hz": summary(right_rates),
        "ipsilateral_DNa02_hz": summary(ipsilateral),
        "turn": summary(turns),
        "turn_magnitude": summary(np.abs(turns).tolist()),
        "direction_accuracy": round(float(np.mean(correct)), 6),
        "whole_brain_rate_hz_per_neuron": summary(network_rates),
        "peak_fraction_spiking_per_step": summary(peak_fractions),
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
    train_strength = 0.35
    test_strength = 0.35
    warmup_steps = 50
    stimulus_steps = 50
    learning_rate = 1.0
    results = {}

    for condition_index, condition in enumerate(CONDITIONS):
        scaler.restore()
        multipliers = np.ones(len(edge_positions), dtype=np.float32)
        history = []
        for epoch in range(epochs):
            epoch_eligibility = np.zeros(len(edge_positions), dtype=np.float32)
            epoch_metrics = []
            for side_index, side in enumerate(SIDES):
                eligibility, metrics = training_episode(
                    brain=brain,
                    stimulus=stimuli[side],
                    dnal=dnal,
                    dnar=dnar,
                    edge_pre=edge_pre,
                    edge_post=edge_post,
                    expected_sign=-1 if side == "left" else 1,
                    condition=condition,
                    seed=train_seed + condition_index * 10000 + epoch * 100 + side_index,
                    strength=train_strength,
                    warmup_steps=warmup_steps,
                    stimulus_steps=stimulus_steps,
                )
                epoch_eligibility += eligibility
                epoch_metrics.append(metrics)
            if condition != "frozen":
                multipliers += learning_rate * epoch_eligibility
                multipliers = np.clip(multipliers, 0.5, 1.5)
                weights = scaler.base_weights.copy()
                weights[edge_positions] = base_selected * multipliers
                scaler.install(weights)
            history.append({
                "epoch": epoch + 1,
                "mean_reward": round(float(np.mean([m["mean_reward"] for m in epoch_metrics])), 6),
                "training_direction_accuracy": round(float(np.mean([m["direction_accuracy"] for m in epoch_metrics])), 6),
                "training_margin_hz": round(float(np.mean([m["mean_margin_hz"] for m in epoch_metrics])), 6),
                "mean_weight_multiplier": round(float(multipliers.mean()), 6),
                "std_weight_multiplier": round(float(multipliers.std()), 6),
            })

        sides = {
            side: evaluate_side(
                brain=brain,
                config=config,
                stimulus=stimuli[side],
                dnal=dnal,
                dnar=dnar,
                expected_sign=-1 if side == "left" else 1,
                seeds=test_seeds,
                strength=test_strength,
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
                "fraction_changed": round(float(np.mean(np.abs(multipliers - 1.0) > 1e-6)), 6),
            },
            "sides": sides,
        }

    checksum_after = sha256_file(weights_path)
    if checksum_before != checksum_after:
        raise RuntimeError("Original weights.npz changed during plasticity experiment")

    def condition_metric(condition: str, metric: str) -> float:
        return float(np.mean([
            results[condition]["sides"][side][metric]["mean"]
            for side in SIDES
        ]))

    frozen_turn = condition_metric("frozen", "turn_magnitude")
    sham_turn = condition_metric("sham_reward", "turn_magnitude")
    random_turn = condition_metric("random_reward", "turn_magnitude")
    contingent_turn = condition_metric("contingent_reward", "turn_magnitude")
    frozen_DNa = condition_metric("frozen", "ipsilateral_DNa02_hz")
    contingent_DNa = condition_metric("contingent_reward", "ipsilateral_DNa02_hz")
    peak = max(
        results["contingent_reward"]["sides"][side]["peak_fraction_spiking_per_step"]["mean"]
        for side in SIDES
    )
    sham_equivalent = abs(sham_turn - frozen_turn) <= 0.03
    random_control_not_equal_to_contingent = contingent_turn - random_turn >= 0.03
    learned_effect = contingent_turn - frozen_turn >= 0.05 and contingent_DNa - frozen_DNa >= 0.5
    network_stable = peak < 0.10
    quality = {
        "original_weights_unchanged": checksum_before == checksum_after,
        "frozen_and_sham_equivalent": sham_equivalent,
        "random_reward_control_below_contingent": random_control_not_equal_to_contingent,
        "contingent_reward_improves_turn_and_DNa02": learned_effect,
        "contingent_network_not_globally_saturated": network_stable,
    }
    quality["experiment_passed"] = all(quality.values())
    report = {
        "schema_version": 1,
        "experiment": "Localized reward-modulated pair-timing plasticity on LC10a outgoing synapses",
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
            "train_strength": train_strength,
            "test_strength": test_strength,
            "warmup_seconds": warmup_steps * brain.dt,
            "stimulus_seconds": stimulus_steps * brain.dt,
            "learning_rate": learning_rate,
            "eligibility": "one-step pre-before-post minus 0.5 times post-before-pre coactivity",
            "reward": "correct direction +1, incorrect -1; random control is balanced +/-1",
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
            "turn_magnitude": {
                "frozen": round(frozen_turn, 6),
                "sham_reward": round(sham_turn, 6),
                "random_reward": round(random_turn, 6),
                "contingent_reward": round(contingent_turn, 6),
            },
            "ipsilateral_DNa02_hz": {
                "frozen": round(frozen_DNa, 6),
                "contingent_reward": round(contingent_DNa, 6),
            },
        },
        "quality_checks": quality,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limits": [
            "This is a deliberately local pair-timing eligibility rule, not evidence that the real LC10a pathway uses this exact plasticity mechanism.",
            "Reward is an experimenter-defined task signal applied after each episode.",
            "Only LC10a outgoing weights change; the Decoder remains frozen.",
            "Weights are bounded and exist only in process memory; original connectome files are never modified.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled reward-modulated plasticity on LC10a outputs")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=11101)
    parser.add_argument("--test-seeds", default="12101,12203,12307")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "reward_plasticity.json",
    )
    args = parser.parse_args()
    test_seeds = [int(value) for value in args.test_seeds.split(",") if value]
    if args.batch < 1 or args.epochs < 1 or not test_seeds:
        parser.error("batch, epochs and test seeds must be non-empty")
    report = run_experiment(
        args.device, args.batch, args.epochs, args.train_seed, test_seeds, args.output
    )
    print(f"saved {args.output}")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(json.dumps(report["quality_checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
