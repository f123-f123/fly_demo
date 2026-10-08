from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
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
from flybrain.reservoir import Readout, auc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DecoderConfig
from app.decoder import DecodeResult, RuleDecoder
from app.encoder import FeatureEncoder
from app.world import World
from run_weight_intervention import batches, population_counts, sha256_file


SCENARIOS = ("null", "target_left", "target_right", "loom_left", "loom_right")
WINDOW_STEPS = (5, 9, 13, 20, 30)
OUTPUT_NAMES = ("steer_left", "steer_right", "escape_left", "escape_right")


def collect_dataset(
    brain: FlyBrain,
    seeds: list[int],
    steps: int,
    warmup_steps: int,
    strength: float,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], int]:
    inputs = {
        "target_left": brain.cells(["LC10a"], side="L"),
        "target_right": brain.cells(["LC10a"], side="R"),
        "loom_left": brain.cells(["LC4", "LPLC2"], side="L"),
        "loom_right": brain.cells(["LC4", "LPLC2"], side="R"),
    }
    outputs = {
        "steer_left": brain.cells(["DNa02"], side="L"),
        "steer_right": brain.cells(["DNa02"], side="R"),
        "escape_left": brain.cells(["DNp01"], side="L"),
        "escape_right": brain.cells(["DNp01"], side="R"),
    }
    descending = brain.cells(["descending_neuron"])
    empty = [name for name, population in {**inputs, **outputs, "descending": descending}.items() if len(population) == 0]
    if empty:
        raise RuntimeError(f"Required populations not found: {', '.join(empty)}")

    slot = np.full(brain.n, -1, dtype=np.int32)
    slot[descending] = np.arange(len(descending), dtype=np.int32)
    trace_decay = np.float32(np.exp(-brain.dt / 0.1))
    records: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        injection = () if scenario == "null" else ((inputs[scenario], strength),)
        for seed in seeds:
            brain.reset(seed)
            for _ in range(warmup_steps):
                brain.step()
            named = np.zeros((brain.batch, steps, len(OUTPUT_NAMES)), dtype=np.uint8)
            trace = np.zeros((len(descending), brain.batch), dtype=np.float32)
            for step in range(steps):
                fired_by_fly = batches(brain.step(inject=injection), brain.batch)
                trace *= trace_decay
                for batch_index, fired in enumerate(fired_by_fly):
                    slots = slot[fired]
                    slots = slots[slots >= 0]
                    trace[slots, batch_index] += 1.0
                for output_index, name in enumerate(OUTPUT_NAMES):
                    named[:, step, output_index] = population_counts(
                        fired_by_fly, outputs[name]
                    )
            for batch_index in range(brain.batch):
                records.append({
                    "scenario": scenario,
                    "seed": seed,
                    "parallel_index": batch_index,
                    "named_spikes": named[batch_index],
                    "descending_trace": trace[:, batch_index].copy(),
                })
    return records, outputs, len(descending)


def decode_rule_samples(
    records: list[dict[str, object]],
    config: DecoderConfig,
    window_steps: int,
    perturbation: str = "none",
    level: float = 0.0,
    random_seed: int = 9001,
) -> dict[str, float]:
    steering_correct: list[float] = []
    steering_latency: list[float] = []
    looming_triggered: list[float] = []
    looming_latency: list[float] = []
    null_false_trigger: list[float] = []
    censor = records[0]["named_spikes"].shape[0] * config.dt + config.publish_steps * config.dt

    for record_index, record in enumerate(records):
        scenario = str(record["scenario"])
        spikes = np.asarray(record["named_spikes"], dtype=np.float32).copy()
        rng = np.random.default_rng(random_seed + record_index)
        if perturbation == "spike_noise" and level > 0.0:
            false_spikes = rng.random(spikes.shape) < min(level * config.dt, 1.0)
            spikes = np.maximum(spikes, false_spikes)
        elif perturbation == "neuron_loss" and level > 0.0:
            alive = rng.random(spikes.shape[1]) >= level
            spikes *= alive[None, :]

        drift = np.zeros(spikes.shape[1], dtype=float)
        if perturbation == "baseline_drift" and level > 0.0:
            drift[int(rng.integers(0, len(drift)))] = level

        decoder_config = copy.deepcopy(config)
        decoder_config.window_steps = window_steps
        decoder = RuleDecoder(decoder_config)
        final_turn = 0.0
        first_correct = math.nan
        first_action = math.nan
        action_seen = False
        for step in range(config.publish_steps - 1, len(spikes), config.publish_steps):
            start = max(0, step - window_steps + 1)
            rates_array = spikes[start:step + 1].sum(axis=0) / ((step - start + 1) * config.dt)
            rates_array += drift
            rates = dict(zip(OUTPUT_NAMES, rates_array.tolist()))
            decoded = decoder.decode(rates, config.publish_steps * config.dt)
            final_turn = decoded.filtered_turn
            now = (step + 1) * config.dt
            expected = -1 if scenario == "target_left" else 1
            if scenario.startswith("target_") and np.isnan(first_correct):
                if final_turn * expected > 0.0:
                    first_correct = now
            if decoded.action:
                action_seen = True
                if np.isnan(first_action):
                    first_action = now

        if scenario.startswith("target_"):
            expected = -1 if scenario == "target_left" else 1
            steering_correct.append(float(final_turn * expected > 0.0))
            steering_latency.append(censor if np.isnan(first_correct) else first_correct)
        elif scenario.startswith("loom_"):
            looming_triggered.append(float(action_seen))
            looming_latency.append(censor if np.isnan(first_action) else first_action)
        else:
            null_false_trigger.append(float(action_seen))

    return {
        "steering_direction_accuracy": float(np.mean(steering_correct)),
        "steering_latency_or_censor_seconds": float(np.mean(steering_latency)),
        "looming_true_trigger_rate": float(np.mean(looming_triggered)),
        "looming_latency_or_censor_seconds": float(np.mean(looming_latency)),
        "null_false_trigger_rate": float(np.mean(null_false_trigger)),
        "latency_censor_seconds": float(censor),
    }


def linear_readout_results(
    train_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
) -> dict[str, object]:
    def fit_task(positive: set[str], negative: set[str]) -> dict[str, object]:
        train = [r for r in train_records if r["scenario"] in positive | negative]
        test = [r for r in test_records if r["scenario"] in positive | negative]
        X_train = np.stack([r["descending_trace"] for r in train])
        y_train = np.asarray([int(r["scenario"] in positive) for r in train])
        groups = np.asarray([r["seed"] for r in train])
        X_test = np.stack([r["descending_trace"] for r in test])
        y_test = np.asarray([int(r["scenario"] in positive) for r in test])
        readout = Readout.fit(
            X_train,
            y_train,
            kind="logistic",
            groups=groups,
            components=(5, 20, 40),
            lambdas=(0.01, 0.1, 1.0, 10.0),
        )
        probability = np.asarray(readout.predict(X_test))
        predicted = probability >= 0.5
        positive_mask = y_test == 1
        negative_mask = ~positive_mask
        return {
            "train_samples": len(train),
            "held_out_samples": len(test),
            "components": readout.components,
            "lambda": readout.lam,
            "cross_validated_auc": round(float(readout.cv_score), 6),
            "held_out_auc": round(float(auc(y_test, probability)), 6),
            "held_out_accuracy": round(float(np.mean(predicted == y_test)), 6),
            "held_out_true_positive_rate": round(float(np.mean(predicted[positive_mask])), 6),
            "held_out_false_positive_rate": round(float(np.mean(predicted[negative_mask])), 6),
        }

    return {
        "steering_right_vs_left": fit_task({"target_right"}, {"target_left"}),
        "looming_vs_null": fit_task({"loom_left", "loom_right"}, {"null"}),
    }


def closed_loop_avoidance(
    brain: FlyBrain,
    base_config: DecoderConfig,
    populations: dict[str, np.ndarray],
    seeds: list[int],
    window_steps: int,
) -> dict[str, object]:
    config = copy.deepcopy(base_config)
    config.window_steps = window_steps
    encoder = FeatureEncoder()
    avoided = 0
    contacts = 0
    latencies: list[float] = []

    for trial_index, seed in enumerate(seeds):
        brain.reset(seed)
        for _ in range(round(1.0 / config.dt)):
            brain.step()
        world = World(seed=seed)
        world._next_threat_time = math.inf
        decoder = RuleDecoder(config)
        buffers = {name: deque(maxlen=window_steps) for name in OUTPUT_NAMES}
        decoded = DecodeResult(config.fixed_speed, 0.0, 0.0, False, {
            name: 0.0 for name in OUTPUT_NAMES
        })

        def advance() -> None:
            nonlocal decoded
            sensors = world.sense(config.dt)
            stimulus = encoder.encode(sensors)
            injection = []
            for name, strength in (
                ("lc10a_left", stimulus.lc10a_left),
                ("lc10a_right", stimulus.lc10a_right),
                ("loom_left", stimulus.loom_left),
                ("loom_right", stimulus.loom_right),
                ("odor_left", stimulus.odor_left),
                ("odor_right", stimulus.odor_right),
            ):
                if strength > 0.0:
                    injection.append((inputs[name], strength))
            fired_by_fly = batches(brain.step(inject=injection), brain.batch)
            fired = fired_by_fly[0]
            for name in OUTPUT_NAMES:
                buffers[name].append(int(np.isin(fired, populations[name]).sum()))
            step_number = int(round(world.time / config.dt)) + 1
            if step_number % config.publish_steps == 0:
                rates = {
                    name: sum(buffer) / len(populations[name]) / (len(buffer) * config.dt)
                    for name, buffer in buffers.items()
                }
                decoded = decoder.decode(rates, config.publish_steps * config.dt)
            world.update(
                config.dt,
                speed=decoded.speed,
                turn=decoded.filtered_turn,
                max_turn_rate=config.max_turn_rate,
                action=decoded.action,
                sensors=sensors,
            )

        inputs = {
            "lc10a_left": brain.cells(["LC10a"], side="L"),
            "lc10a_right": brain.cells(["LC10a"], side="R"),
            "loom_left": brain.cells(["LC4", "LPLC2"], side="L"),
            "loom_right": brain.cells(["LC4", "LPLC2"], side="R"),
            "odor_left": np.concatenate([
                brain.cells(["ORN_DM1"], side="L"), brain.cells(["ORN_VA2"], side="L")
            ]),
            "odor_right": np.concatenate([
                brain.cells(["ORN_DM1"], side="R"), brain.cells(["ORN_VA2"], side="R")
            ]),
        }
        for _ in range(round(2.0 / config.dt)):
            advance()
        world.spawn_obstacle(side=(-1.0, 0.0, 1.0)[trial_index % 3], distance=7.0)
        spawned_at = world.time
        first_action = math.nan
        for _ in range(round(4.0 / config.dt)):
            advance()
            if decoded.action and np.isnan(first_action):
                first_action = world.time - spawned_at
            if world.statistics()["threat_trials"]:
                break
        stats = world.statistics()
        avoided += int(stats["threats_avoided"])
        contacts += int(stats["threat_contacts"])
        latencies.append(4.1 if np.isnan(first_action) else first_action)

    trials = avoided + contacts
    return {
        "trials": trials,
        "avoided": avoided,
        "contacts": contacts,
        "avoidance_success_rate": round(avoided / trials if trials else 0.0, 6),
        "mean_action_latency_or_censor_seconds": round(float(np.mean(latencies)), 6),
        "action_latency_censor_seconds": 4.1,
    }


def run_experiment(
    device: str,
    batch: int,
    train_seeds: list[int],
    test_seeds: list[int],
    output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    config = DecoderConfig.load()
    weights_path = Path(DATA) / "weights.npz"
    weights_checksum_before = sha256_file(weights_path)
    brain = FlyBrain(
        device=device,
        batch=batch,
        seed=train_seeds[0],
        dt=config.dt,
        sensory_input=False,
    )
    steps = round(1.0 / config.dt)
    warmup_steps = round(1.0 / config.dt)
    train_records, populations, descending_size = collect_dataset(
        brain, train_seeds, steps, warmup_steps, 0.8
    )
    test_records, _, _ = collect_dataset(
        brain, test_seeds, steps, warmup_steps, 0.8
    )

    window_results = {
        str(round(window * config.dt * 1000)): decode_rule_samples(
            test_records, config, window
        )
        for window in WINDOW_STEPS
    }
    noise_results = {
        str(level): decode_rule_samples(
            test_records, config, config.window_steps, "spike_noise", level
        )
        for level in (0.0, 1.0, 3.0, 5.0)
    }
    drift_results = {
        str(level): decode_rule_samples(
            test_records, config, config.window_steps, "baseline_drift", level
        )
        for level in (0.0, 1.0, 3.0, 5.0)
    }
    loss_results = {
        str(level): decode_rule_samples(
            test_records, config, config.window_steps, "neuron_loss", level
        )
        for level in (0.0, 0.1, 0.25, 0.5)
    }
    readout = linear_readout_results(train_records, test_records)

    closed_loop_brain = FlyBrain(
        device=device,
        batch=1,
        seed=test_seeds[0],
        dt=config.dt,
        sensory_input=False,
    )
    closed_loop_populations = {
        "steer_left": closed_loop_brain.cells(["DNa02"], side="L"),
        "steer_right": closed_loop_brain.cells(["DNa02"], side="R"),
        "escape_left": closed_loop_brain.cells(["DNp01"], side="L"),
        "escape_right": closed_loop_brain.cells(["DNp01"], side="R"),
    }
    avoidance_seeds = [6101, 6203, 6301, 6401, 6503]
    avoidance = {
        str(round(window * config.dt * 1000)): closed_loop_avoidance(
            closed_loop_brain,
            config,
            closed_loop_populations,
            avoidance_seeds,
            window,
        )
        for window in WINDOW_STEPS
    }

    weights_checksum_after = sha256_file(weights_path)
    if weights_checksum_before != weights_checksum_after:
        raise RuntimeError("Original weights.npz changed during robustness experiment")
    default = window_results[str(round(config.window_steps * config.dt * 1000))]
    quality = {
        "original_weights_unchanged": True,
        "default_rule_steering_accuracy_at_least_95_percent": default["steering_direction_accuracy"] >= 0.95,
        "default_rule_looming_trigger_at_least_95_percent": default["looming_true_trigger_rate"] >= 0.95,
        "default_rule_false_trigger_at_most_5_percent": default["null_false_trigger_rate"] <= 0.05,
        "linear_steering_held_out_accuracy_at_least_80_percent": readout["steering_right_vs_left"]["held_out_accuracy"] >= 0.80,
        "linear_looming_held_out_accuracy_at_least_80_percent": readout["looming_vs_null"]["held_out_accuracy"] >= 0.80,
        "default_closed_loop_avoidance_at_least_80_percent": avoidance[str(round(config.window_steps * config.dt * 1000))]["avoidance_success_rate"] >= 0.80,
    }
    quality["experiment_passed"] = all(quality.values())
    report = {
        "schema_version": 1,
        "experiment": "Decoder robustness and held-out linear readout",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provenance": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "flybrain_version": importlib.metadata.version("flybrain"),
            "device": brain.device,
            "weights_sha256_before": weights_checksum_before,
            "weights_sha256_after": weights_checksum_after,
            "decoder_sha256": sha256_file(ROOT / "config" / "decoder.json"),
            "decoder_parameters": asdict(config),
        },
        "protocol": {
            "scenarios": list(SCENARIOS),
            "train_seeds": train_seeds,
            "held_out_test_seeds": test_seeds,
            "parallel_flies_per_seed": batch,
            "samples_per_scenario_train": len(train_seeds) * batch,
            "samples_per_scenario_test": len(test_seeds) * batch,
            "stimulus_seconds": steps * config.dt,
            "stimulus_strength": 0.8,
            "window_milliseconds": [round(window * config.dt * 1000) for window in WINDOW_STEPS],
            "avoidance_seeds_per_window": avoidance_seeds,
        },
        "population_comparison": {
            "named_rule_output_neurons": {name: len(population) for name, population in populations.items()},
            "descending_population_features": descending_size,
            "constraint": "DNa02 and DNp01 contain one neuron per side, so within-type population-vs-single-neuron averaging is not defined. The valid comparison is the named bilateral rule versus a linear readout over all descending neurons.",
        },
        "rule_decoder": {
            "window_sweep": window_results,
            "spike_noise_hz": noise_results,
            "baseline_drift_hz": drift_results,
            "random_named_neuron_loss_probability": loss_results,
        },
        "linear_readout": readout,
        "closed_loop_avoidance_by_window_ms": avoidance,
        "quality_checks": quality,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limits": [
            "Spike noise adds false spikes independently; baseline drift adds a constant rate to one random named output channel per trial.",
            "Random neuron loss is severe because each named output contains one neuron per side.",
            "The linear Readout is trained only on frozen FlyBrain activity and never changes connectome weights.",
            "Closed-loop avoidance includes the documented non-neural escape-direction policy after DNp01 authorizes action.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FlyBrain Decoder robustness")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--train-seeds", default="4101,4201,4301")
    parser.add_argument("--test-seeds", default="5101,5201")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "decoder_robustness.json",
    )
    args = parser.parse_args()
    train_seeds = [int(value) for value in args.train_seeds.split(",") if value]
    test_seeds = [int(value) for value in args.test_seeds.split(",") if value]
    if args.batch < 1 or not train_seeds or not test_seeds:
        parser.error("batch and both seed lists must be non-empty")
    report = run_experiment(args.device, args.batch, train_seeds, test_seeds, args.output)
    print(f"saved {args.output}")
    print(json.dumps(report["quality_checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
