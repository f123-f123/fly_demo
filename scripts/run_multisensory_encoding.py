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

from run_weight_intervention import batches, population_counts, sha256_file


CHANNELS = {
    "warming": {
        "input": ["TRN_VP2"],
        "output": ["VP2_l2PN", "VP1m+VP2_lvPN1", "VP2+VC5_l2PN", "VP2+_adPN", "VP2_adPN"],
        "input_role": "thermoreceptor sensory neurons",
        "output_role": "VP2 projection neurons",
    },
    "cooling": {
        "input": ["TRN_VP3a", "TRN_VP3b"],
        "output": ["VP3+_l2PN", "VP3+_vPN", "VP1l+VP3_ilPN", "VP3+VP1l_ivPN"],
        "input_role": "thermoreceptor sensory neurons",
        "output_role": "VP3 projection neurons",
    },
    "drying": {
        "input": ["HRN_VP4"],
        "output": ["VP4+_vPN", "VP4_vPN", "VP4+VL1_l2PN"],
        "input_role": "hygroreceptor sensory neurons",
        "output_role": "VP4 projection neurons",
    },
    "moistening": {
        "input": ["HRN_VP5"],
        "output": ["VP5+_l2PN,VP5+VP2_l2PN", "VP5+VP3_l2PN", "VP1m+VP5_ilPN", "VP5+Z_adPN"],
        "input_role": "hygroreceptor sensory neurons",
        "output_role": "VP5 projection neurons",
    },
    "light": {
        "input": ["OCG01a", "OCG01b", "OCG01c", "OCG01d", "OCG01e", "OCG01f", "OCG02a", "OCG02b", "OCG02c"],
        "output": ["DNp20", "DNp22", "OCC02b"],
        "input_role": "ocellar projection proxy; raw ocellar photoreceptors are not identified in this dataset",
        "output_role": "strong annotated downstream targets, not a validated light motor pathway",
    },
}
ENCODERS = ("intensity", "rate_of_change", "spike_probability", "adaptive_threshold")
LEVELS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


def drive_for_step(
    encoder: str,
    levels: np.ndarray,
    step: int,
    rng: np.random.Generator,
    adaptation: np.ndarray,
) -> np.ndarray:
    if encoder == "intensity":
        return levels * 0.8
    if encoder == "rate_of_change":
        return levels * 0.8 if step == 0 else np.zeros_like(levels)
    if encoder == "spike_probability":
        return (rng.random(len(levels)) < levels).astype(np.float32) * 0.8
    if encoder == "adaptive_threshold":
        difference = np.maximum(levels - adaptation, 0.0)
        adaptation += 0.18 * (levels - adaptation)
        return np.clip(difference * 1.6, 0.0, 0.8)
    raise ValueError(f"Unknown encoder: {encoder}")


def summarize(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std()), 6),
        "values": np.round(values, 6).tolist(),
    }


def run_condition(
    brain: FlyBrain,
    input_population: np.ndarray,
    output_populations: dict[str, np.ndarray],
    encoder: str,
    seed: int,
    trials_per_level: int,
    warmup_steps: int,
    stimulus_steps: int,
) -> dict[str, dict[str, np.ndarray]]:
    expanded_levels = np.repeat(LEVELS, trials_per_level)
    brain.reset(seed)
    for _ in range(warmup_steps):
        brain.step()
    rng = np.random.default_rng(seed + 70001)
    adaptation = np.zeros(brain.batch, dtype=np.float32)
    counts = {
        "input": np.zeros(brain.batch, dtype=np.int32),
        **{name: np.zeros(brain.batch, dtype=np.int32) for name in output_populations},
    }
    first_output = np.full(brain.batch, np.nan, dtype=float)
    intended_name = next(iter(output_populations))
    for step in range(stimulus_steps):
        amount = drive_for_step(encoder, expanded_levels, step, rng, adaptation)
        fired_by_fly = batches(
            brain.step(inject=((input_population, amount),)), brain.batch
        )
        counts["input"] += population_counts(fired_by_fly, input_population)
        for name, population in output_populations.items():
            step_counts = population_counts(fired_by_fly, population)
            counts[name] += step_counts
            if name == intended_name:
                newly_fired = (step_counts > 0) & np.isnan(first_output)
                first_output[newly_fired] = (step + 1) * brain.dt

    seconds = stimulus_steps * brain.dt
    rates = {
        "input": counts["input"] / len(input_population) / seconds,
        **{
            name: counts[name] / len(population) / seconds
            for name, population in output_populations.items()
        },
    }
    latency_censor = seconds + brain.dt
    latency = np.where(np.isfinite(first_output), first_output, latency_censor)
    result: dict[str, dict[str, np.ndarray]] = {}
    for level_index, level in enumerate(LEVELS):
        selection = slice(
            level_index * trials_per_level,
            (level_index + 1) * trials_per_level,
        )
        result[str(float(level))] = {
            **{name: values[selection] for name, values in rates.items()},
            "intended_output_latency_or_censor_seconds": latency[selection],
        }
    return result


def compile_channel(
    raw_by_encoder: dict[str, list[dict[str, dict[str, np.ndarray]]]],
    channel: str,
) -> dict[str, object]:
    compiled: dict[str, object] = {}
    output_names = list(CHANNELS)
    for encoder, seed_runs in raw_by_encoder.items():
        levels: dict[str, object] = {}
        for level in map(str, LEVELS.astype(float)):
            keys = seed_runs[0][level]
            levels[level] = {
                key: summarize(np.concatenate([run[level][key] for run in seed_runs]))
                for key in keys
            }
        intended_curve = [levels[str(float(level))][channel]["mean"] for level in LEVELS]
        input_curve = [levels[str(float(level))]["input"]["mean"] for level in LEVELS]
        baseline_values = np.concatenate([run["0.0"][channel] for run in seed_runs])
        maximum_values = np.concatenate([run["1.0"][channel] for run in seed_runs])
        signal = float(maximum_values.mean() - baseline_values.mean())
        snr = signal / max(float(baseline_values.std()), 0.1)
        off_target_deltas = []
        for other in output_names:
            if other == channel:
                continue
            baseline = levels["0.0"][other]["mean"]
            maximum = levels["1.0"][other]["mean"]
            off_target_deltas.append(max(0.0, maximum - baseline))
        separation = signal / max(max(off_target_deltas, default=0.0), 0.1)
        per_seed_maximum = [float(run["1.0"][channel].mean()) for run in seed_runs]
        reproducibility_cv = float(np.std(per_seed_maximum) / max(abs(np.mean(per_seed_maximum)), 0.1))
        compiled[encoder] = {
            "levels": levels,
            "metrics": {
                "input_curve_hz": np.round(input_curve, 6).tolist(),
                "intended_output_curve_hz": np.round(intended_curve, 6).tolist(),
                "intended_output_signal_hz": round(signal, 6),
                "intended_output_snr": round(snr, 6),
                "channel_separation_ratio": round(separation, 6),
                "reproducibility_cv_across_seeds": round(reproducibility_cv, 6),
                "intended_curve_monotonic_non_decreasing": bool(
                    all(a <= b + 0.25 for a, b in zip(intended_curve, intended_curve[1:]))
                ),
                "mean_maximum_level_latency_or_censor_seconds": levels["1.0"]["intended_output_latency_or_censor_seconds"]["mean"],
            },
        }
    return compiled


def run_experiment(
    device: str,
    trials_per_level: int,
    seeds: list[int],
    output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    weights_path = Path(DATA) / "weights.npz"
    checksum_before = sha256_file(weights_path)
    brain = FlyBrain(
        device=device,
        batch=trials_per_level * len(LEVELS),
        seed=seeds[0],
        dt=0.02,
        sensory_input=False,
    )
    channel_inputs = {
        name: brain.cells(spec["input"])
        for name, spec in CHANNELS.items()
    }
    channel_outputs = {
        name: brain.cells(spec["output"])
        for name, spec in CHANNELS.items()
    }
    empty = [
        name for name, population in {**channel_inputs, **{f"{k}_output": v for k, v in channel_outputs.items()}}.items()
        if len(population) == 0
    ]
    if empty:
        raise RuntimeError(f"Required populations not found: {', '.join(empty)}")

    warmup_steps = 50
    stimulus_steps = 50
    channels: dict[str, object] = {}
    for channel_index, channel in enumerate(CHANNELS):
        raw_by_encoder: dict[str, list[dict[str, dict[str, np.ndarray]]]] = {}
        ordered_outputs = {
            channel: channel_outputs[channel],
            **{
                other: channel_outputs[other]
                for other in CHANNELS
                if other != channel
            },
        }
        for encoder_index, encoder in enumerate(ENCODERS):
            raw_by_encoder[encoder] = [
                run_condition(
                    brain,
                    channel_inputs[channel],
                    ordered_outputs,
                    encoder,
                    seed + channel_index * 1000 + encoder_index * 100,
                    trials_per_level,
                    warmup_steps,
                    stimulus_steps,
                )
                for seed in seeds
            ]
        channels[channel] = {
            "population_sizes": {
                "input": len(channel_inputs[channel]),
                "intended_output": len(channel_outputs[channel]),
            },
            "input_types": CHANNELS[channel]["input"],
            "output_types": CHANNELS[channel]["output"],
            "input_role": CHANNELS[channel]["input_role"],
            "output_role": CHANNELS[channel]["output_role"],
            "encoders": compile_channel(raw_by_encoder, channel),
        }

    checksum_after = sha256_file(weights_path)
    if checksum_before != checksum_after:
        raise RuntimeError("Original weights.npz changed during sensory encoding experiment")
    channel_quality = {}
    for channel, data in channels.items():
        encoder_metrics = {
            encoder: result["metrics"]
            for encoder, result in data["encoders"].items()
        }
        channel_quality[channel] = {
            "at_least_one_encoder_snr_at_least_3": any(
                metrics["intended_output_snr"] >= 3.0 for metrics in encoder_metrics.values()
            ),
            "at_least_one_encoder_separation_at_least_1": any(
                metrics["channel_separation_ratio"] >= 1.0 for metrics in encoder_metrics.values()
            ),
            "at_least_one_encoder_cv_at_most_0.25": any(
                metrics["reproducibility_cv_across_seeds"] <= 0.25 for metrics in encoder_metrics.values()
            ),
        }
    quality = {
        "original_weights_unchanged": checksum_before == checksum_after,
        "channels": channel_quality,
    }
    quality["experiment_passed"] = bool(
        quality["original_weights_unchanged"]
        and all(all(checks.values()) for checks in channel_quality.values())
    )
    report = {
        "schema_version": 1,
        "experiment": "Software temperature, humidity and light-proxy encoding",
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
            "encoders": list(ENCODERS),
            "normalized_signal_levels": LEVELS.tolist(),
            "seeds": seeds,
            "trials_per_level_per_seed": trials_per_level,
            "samples_per_curve_point": trials_per_level * len(seeds),
            "warmup_seconds": warmup_steps * brain.dt,
            "stimulus_seconds": stimulus_steps * brain.dt,
            "dt_seconds": brain.dt,
            "movement_decoder_used": False,
        },
        "channels": channels,
        "quality_checks": quality,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limits": [
            "Normalized software levels are not yet calibrated to degrees Celsius, relative humidity, or lux.",
            "The OCG light input is an ocellar projection proxy because raw ocellar photoreceptors were not identified in the installed annotations.",
            "No sensory result is connected to movement; downstream descending-neuron scans are a separate phase.",
            "SNR uses a 0.1 Hz floor when baseline variance is near zero.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare software sensory encoders in FlyBrain")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--trials-per-level", type=int, default=8)
    parser.add_argument("--seeds", default="7101,7203,7307")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "multisensory_encoding.json",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if args.trials_per_level < 1 or not seeds:
        parser.error("trials-per-level and seeds must be non-empty")
    report = run_experiment(args.device, args.trials_per_level, seeds, args.output)
    print(f"saved {args.output}")
    print(json.dumps(report["quality_checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
