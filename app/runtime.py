from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict
from typing import Awaitable, Callable

from .brain import BrainRuntime
from .config import DecoderConfig
from .decoder import DecodeResult, RuleDecoder
from .encoder import FeatureEncoder, Stimulus
from .logging import SessionLogger
from .world import Sensors, World


StateCallback = Callable[[dict[str, object]], Awaitable[None]]


class DemoRuntime:
    def __init__(
        self,
        config: DecoderConfig,
        device: str = "auto",
        seed: int = 1,
    ):
        self.config = config
        self.seed = seed
        self.brain = BrainRuntime(config, device=device, seed=seed)
        self.world = World(seed=seed)
        self.encoder = FeatureEncoder()
        self.decoder = RuleDecoder(config)
        self.logger = SessionLogger()
        self.paused = False
        self.running = False
        self.generation = 0
        self.step_count = 0
        self.sequence = 0
        self.manual_kind: str | None = None
        self.manual_until = 0.0
        self.last_stimulus = Stimulus()
        self.last_sensors = self.world.sense(config.dt)
        self.last_rates = {name: 0.0 for name in self.brain.outputs}
        self.last_total_spikes = 0
        self.last_descending_spikes = 0
        self.last_decode = DecodeResult(0.0, 0.0, 0.0, False, {
            name: 0.0 for name in self.brain.outputs
        })
        self.realtime_factor = 1.0
        self.step_cost_ms = {}

    def initialize(self) -> None:
        self.brain.warmup()

    def close(self) -> None:
        self.logger.close()

    def reset(self) -> None:
        self.generation += 1
        self.step_count = 0
        self.sequence = 0
        self.manual_kind = None
        self.manual_until = 0.0
        self.world.reset()
        self.decoder.reset()
        self.brain.reset(self.seed)
        self.brain.warmup()
        self.last_stimulus = Stimulus()
        self.last_sensors = self.world.sense(self.config.dt)
        self.last_rates = {name: 0.0 for name in self.brain.outputs}
        self.last_total_spikes = 0
        self.last_descending_spikes = 0
        self.last_decode = DecodeResult(0.0, 0.0, 0.0, False, {
            name: 0.0 for name in self.brain.outputs
        })

    def set_manual_stimulus(self, kind: str, duration: float = 1.0) -> None:
        valid = {"target_left", "target_right", "loom_left", "loom_right", "odor_left", "odor_right"}
        if kind not in valid:
            raise ValueError(f"Unknown manual stimulus: {kind}")
        self.manual_kind = kind
        self.manual_until = self.world.time + max(0.1, min(duration, 5.0))

    def _manual_stimulus(self) -> Stimulus | None:
        if self.manual_kind is None or self.world.time >= self.manual_until:
            self.manual_kind = None
            return None
        stimulus = Stimulus(source=f"manual:{self.manual_kind}")
        if self.manual_kind == "target_left":
            stimulus.lc10a_left = 0.8
        elif self.manual_kind == "target_right":
            stimulus.lc10a_right = 0.8
        elif self.manual_kind == "loom_left":
            stimulus.loom_left = 0.8
        elif self.manual_kind == "loom_right":
            stimulus.loom_right = 0.8
        elif self.manual_kind == "odor_left":
            stimulus.odor_left = 0.8
        elif self.manual_kind == "odor_right":
            stimulus.odor_right = 0.8
        return stimulus

    def step(self) -> dict[str, object] | None:
        started = time.perf_counter()
        dt = self.config.dt
        sensors = self.world.sense(dt)
        stimulus = self._manual_stimulus() or self.encoder.encode(sensors)
        sensed = time.perf_counter()
        fired, rates, descending_spikes = self.brain.step(stimulus)
        simulated = time.perf_counter()

        self.step_count += 1
        self.last_sensors = sensors
        self.last_stimulus = stimulus
        self.last_rates = rates
        self.last_total_spikes = len(fired)
        self.last_descending_spikes = descending_spikes

        if self.step_count % self.config.publish_steps == 0:
            self.last_decode = self.decoder.decode(
                rates,
                elapsed_seconds=self.config.publish_steps * dt,
            )

        self.world.update(
            dt,
            speed=self.last_decode.speed,
            turn=self.last_decode.filtered_turn,
            max_turn_rate=self.config.max_turn_rate,
            action=self.last_decode.action,
            sensors=sensors,
        )
        updated = time.perf_counter()
        self.step_cost_ms = {
            "sense": round((sensed - started) * 1000, 3),
            "brain": round((simulated - sensed) * 1000, 3),
            "motion": round((updated - simulated) * 1000, 3),
            "total": round((updated - started) * 1000, 3),
        }

        if self.step_count % self.config.publish_steps != 0:
            return None
        self.sequence += 1
        state = self.snapshot()
        self.logger.write(state)
        return state

    def snapshot(self) -> dict[str, object]:
        sensors = self.last_sensors
        decode = self.last_decode
        rates = self.last_rates
        sensor_values = asdict(sensors)
        for name in ("target_ttc", "obstacle_ttc"):
            if not math.isfinite(sensor_values[name]):
                sensor_values[name] = None
        return {
            "type": "state",
            "seq": self.sequence,
            "sim_time_ms": round(self.world.time * 1000),
            "system": {
                "device": self.brain.device,
                "seed": self.seed,
                "paused": self.paused,
                "neural_hz": round(1.0 / self.config.dt),
                "control_hz": round(1.0 / (self.config.dt * self.config.publish_steps)),
                "window_ms": round(self.config.window_steps * self.config.dt * 1000),
                "realtime_factor": round(self.realtime_factor, 2),
                "step_cost_ms": self.step_cost_ms,
                "log_file": self.logger.path.name,
                "log_capped": self.logger.capped,
            },
            "world": {
                "fly": asdict(self.world.fly),
                "target": asdict(self.world.target),
                "obstacle": asdict(self.world.obstacle),
                "food_sources": [asdict(source) for source in self.world.food_sources],
                "colliders": [asdict(collider) for collider in self.world.colliders],
                "wind_zones": [asdict(zone) for zone in self.world.wind_zones],
                "wind": {"x": self.world.wind[0], "y": self.world.wind[1], "z": self.world.wind[2]},
                "motion_policy": self.world.motion_policy,
                "stats": self.world.statistics(),
                "arena": {
                    "half_extent": self.world.half_extent,
                    "glass_height": self.world.glass_height,
                    "biome": "temperate_microhabitat",
                    "layout_index": self.world.layout_index,
                    "terrain": self.world.terrain,
                },
                "sensors": {
                    **sensor_values,
                    "target_bearing_degrees": math.degrees(sensors.target_bearing),
                    "target_elevation_degrees": math.degrees(sensors.target_elevation),
                    "obstacle_bearing_degrees": math.degrees(sensors.obstacle_bearing),
                    "obstacle_elevation_degrees": math.degrees(sensors.obstacle_elevation),
                    "angular_size_degrees": math.degrees(sensors.angular_size),
                    "target_angular_size_degrees": math.degrees(sensors.target_angular_size),
                },
            },
            "input": {
                **self.last_stimulus.to_dict(),
                "population_sizes": {
                    name: size
                    for name, size in self.brain.population_sizes().items()
                    if name != "descending"
                },
            },
            "brain": {
                "total_spikes": self.last_total_spikes,
                "descending_spikes": self.last_descending_spikes,
                "descending_population": len(self.brain.descending),
                "rates_hz": {name: round(value, 3) for name, value in rates.items()},
                "step_counts": self.brain.window.last_counts,
                "baseline_hz": self.config.baseline,
                "scale_hz": self.config.scale,
            },
            "decoder": {
                "raw_turn": round(decode.raw_turn, 4),
                "filtered_turn": round(decode.filtered_turn, 4),
                "odor_turn": round(decode.odor_turn, 4),
                "speed": round(decode.speed, 4),
                "action": decode.action,
                "normalized": {
                    name: round(value, 4) for name, value in decode.normalized.items()
                },
                "alpha": self.config.alpha,
                "deadzone": self.config.deadzone,
                "turn_sign": self.config.turn_sign,
                "sources": {
                    "speed": "fixed policy gated by sensory behavior state",
                    "turn": "DNa02 + verified ORN spike difference",
                    "action": "DNp01",
                    "altitude": "correlated exploration / visual surface policy; DNp01 gates escape, direction is non-neural",
                },
            },
        }

    async def run(self, callback: StateCallback) -> None:
        self.running = True
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        real_start = time.perf_counter()
        sim_start = self.world.time
        generation = self.generation
        while self.running:
            if self.paused:
                await asyncio.sleep(0.05)
                deadline = loop.time()
                real_start = time.perf_counter()
                sim_start = self.world.time
                continue

            state = self.step()
            now = time.perf_counter()
            if self.generation != generation:
                generation = self.generation
                real_start = now
                sim_start = self.world.time
            real_elapsed = max(now - real_start, 1e-6)
            self.realtime_factor = (self.world.time - sim_start) / real_elapsed
            if state is not None:
                await callback(state)

            deadline += self.config.dt
            wait = deadline - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            else:
                deadline = loop.time()
                # Even when slower than real time, let startup, sockets and pause run.
                await asyncio.sleep(0)
