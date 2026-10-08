from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.data import DATA

from run_weight_intervention import sha256_file


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(value: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value[key]) for key in ("mean", "std", "min", "max")
    }


def source_hashes() -> dict[str, str]:
    paths: list[Path] = []
    for directory, patterns in (
        (ROOT / "app", ("*.py",)),
        (ROOT / "scripts", ("*.py", "*.ps1")),
        (ROOT / "tests", ("*.py",)),
        (ROOT / "web", ("*.js", "*.css", "*.html")),
    ):
        for pattern in patterns:
            paths.extend(directory.glob(pattern))
    for name in ("requirements.txt", "package.json", "package-lock.json"):
        path = ROOT / name
        if path.exists():
            paths.append(path)
    paths.extend(ROOT.glob("*.ps1"))
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(set(paths))
    }


def run_tests() -> dict[str, object]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", output)
    return {
        "command": " ".join(command[1:]),
        "python_executable": sys.executable,
        "return_code": completed.returncode,
        "tests_run": int(match.group(1)) if match else None,
        "reported_seconds": float(match.group(2)) if match else None,
        "passed": completed.returncode == 0 and "\nOK" in output,
        "output_tail": output[-2000:],
    }


def side_summary(side: dict[str, object], names: tuple[str, ...]) -> dict[str, object]:
    return {name: metric_summary(side[name]) for name in names}


def freeze(output: Path) -> dict[str, object]:
    decoder_path = ROOT / "config" / "decoder.json"
    steering_path = ROOT / "config" / "experiments" / "lc10a_weight_intervention.json"
    escape_path = ROOT / "config" / "experiments" / "looming_weight_intervention.json"
    weights_path = Path(DATA) / "weights.npz"
    decoder = load_json(decoder_path)
    steering = load_json(steering_path)
    escape = load_json(escape_path)
    steering_baseline = steering["conditions"]["baseline"]
    escape_baseline = escape["conditions"]["baseline"]
    tests = run_tests()
    if not tests["passed"]:
        raise RuntimeError("Cannot freeze a baseline while the test suite is failing")

    report = {
        "schema_version": 1,
        "baseline_id": "flyworld-flybrain-static-baseline-2026-09-22",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "freeze_kind": "retrospective manifest over preserved raw baseline conditions",
        "provenance": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "flybrain_version": importlib.metadata.version("flybrain"),
            "weights_path": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "decoder_path": str(decoder_path),
            "decoder_sha256": sha256_file(decoder_path),
            "steering_result_sha256": sha256_file(steering_path),
            "escape_result_sha256": sha256_file(escape_path),
            "source_sha256": source_hashes(),
        },
        "decoder_parameters": decoder,
        "random_seeds": {
            "steering_and_escape_weight_interventions": steering["protocol"]["seeds"],
            "decoder_calibration_seed": decoder.get("calibration", {}).get("seed"),
        },
        "baseline_protocol": {
            "dt_seconds": steering["protocol"]["dt_seconds"],
            "warmup_steps": steering["protocol"]["warmup_steps"],
            "stimulus_steps": steering["protocol"]["stimulus_steps"],
            "stimulus_strength": steering["protocol"]["stimulus_strength"],
            "parallel_flies_per_seed": steering["protocol"]["parallel_flies_per_seed"],
            "trials_per_side": steering["protocol"]["trials_per_side_per_condition"],
        },
        "steering_baseline": {
            "source": str(steering_path.relative_to(ROOT)),
            "condition": "baseline",
            "left": side_summary(steering_baseline["sides"]["left"], (
                "steer_left", "steer_right", "final_turn", "integrated_turn_radians",
                "direction_correct", "whole_brain_spikes",
                "whole_brain_rate_hz_per_neuron", "peak_fraction_spiking_per_step",
            )),
            "right": side_summary(steering_baseline["sides"]["right"], (
                "steer_left", "steer_right", "final_turn", "integrated_turn_radians",
                "direction_correct", "whole_brain_spikes",
                "whole_brain_rate_hz_per_neuron", "peak_fraction_spiking_per_step",
            )),
            "performance": steering_baseline["performance"],
        },
        "escape_baseline": {
            "source": str(escape_path.relative_to(ROOT)),
            "condition": "baseline",
            "left": side_summary(escape_baseline["sides"]["left"], (
                "escape_left", "escape_right", "action_triggered",
                "action_latency_or_censor_seconds", "whole_brain_spikes",
                "whole_brain_rate_hz_per_neuron", "peak_fraction_spiking_per_step",
            )),
            "right": side_summary(escape_baseline["sides"]["right"], (
                "escape_left", "escape_right", "action_triggered",
                "action_latency_or_censor_seconds", "whole_brain_spikes",
                "whole_brain_rate_hz_per_neuron", "peak_fraction_spiking_per_step",
            )),
            "performance": escape_baseline["performance"],
        },
        "test_suite": tests,
        "integrity_notes": [
            "This manifest was created after the causal experiments from their preserved raw baseline conditions; it does not claim an earlier creation timestamp.",
            "The source result hashes bind this summary to the full per-trial values stored in the two experiment JSON files.",
            "All intervention and plasticity experiments report the same original weights checksum and modify weights only in process memory.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the reproducible FlyWorld/FlyBrain baseline manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "experiments" / "baseline_manifest.json",
    )
    args = parser.parse_args()
    report = freeze(args.output)
    print(f"saved {args.output}")
    print(json.dumps({
        "weights_sha256": report["provenance"]["weights_sha256"],
        "decoder_sha256": report["provenance"]["decoder_sha256"],
        "tests_run": report["test_suite"]["tests_run"],
        "tests_passed": report["test_suite"]["passed"],
        "steering_rtf": report["steering_baseline"]["performance"]["realtime_factor"],
        "escape_rtf": report["escape_baseline"]["performance"]["realtime_factor"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
