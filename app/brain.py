from __future__ import annotations

import numpy as np
from flybrain import FlyBrain

from .activity import ActivityWindow
from .config import DecoderConfig
from .encoder import Stimulus


class BrainRuntime:
    def __init__(
        self,
        config: DecoderConfig,
        device: str = "auto",
        seed: int = 1,
    ):
        self.config = config
        self.seed = seed
        self.brain = FlyBrain(
            device=device,
            seed=seed,
            dt=config.dt,
            sensory_input=False,
        )
        self.inputs = {
            "lc10a_left": self.brain.cells(["LC10a"], side="L"),
            "lc10a_right": self.brain.cells(["LC10a"], side="R"),
            "loom_left": self.brain.cells(["LC4", "LPLC2"], side="L"),
            "loom_right": self.brain.cells(["LC4", "LPLC2"], side="R"),
            "odor_left": np.concatenate([
                self.brain.cells(["ORN_DM1"], side="L"),
                self.brain.cells(["ORN_VA2"], side="L"),
            ]),
            "odor_right": np.concatenate([
                self.brain.cells(["ORN_DM1"], side="R"),
                self.brain.cells(["ORN_VA2"], side="R"),
            ]),
        }
        self.outputs = {
            "steer_left": self.brain.cells(["DNa02"], side="L"),
            "steer_right": self.brain.cells(["DNa02"], side="R"),
            "escape_left": self.brain.cells(["DNp01"], side="L"),
            "escape_right": self.brain.cells(["DNp01"], side="R"),
            "odor_left": np.concatenate([
                self.brain.cells(["ORN_DM1"], side="L"),
                self.brain.cells(["ORN_VA2"], side="L"),
            ]),
            "odor_right": np.concatenate([
                self.brain.cells(["ORN_DM1"], side="R"),
                self.brain.cells(["ORN_VA2"], side="R"),
            ]),
        }
        empty = [
            name
            for name, indices in {**self.inputs, **self.outputs}.items()
            if len(indices) == 0
        ]
        if empty:
            raise RuntimeError(f"FlyBrain populations not found: {', '.join(empty)}")
        self.descending = self.brain.cells(["descending_neuron"])
        self.window = ActivityWindow(
            self.outputs,
            dt=config.dt,
            window_steps=config.window_steps,
        )

    @property
    def device(self) -> str:
        return self.brain.device

    def population_sizes(self) -> dict[str, int]:
        groups = {**self.inputs, **self.outputs, "descending": self.descending}
        return {name: len(indices) for name, indices in groups.items()}

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.brain.reset(self.seed)
        self.window.reset()

    def warmup(self, steps: int = 50) -> None:
        for _ in range(steps):
            self.brain.step()
        self.window.reset()

    def step(self, stimulus: Stimulus) -> tuple[np.ndarray, dict[str, float], int]:
        inject = []
        for name, strength in (
            ("lc10a_left", stimulus.lc10a_left),
            ("lc10a_right", stimulus.lc10a_right),
            ("loom_left", stimulus.loom_left),
            ("loom_right", stimulus.loom_right),
            ("odor_left", stimulus.odor_left),
            ("odor_right", stimulus.odor_right),
        ):
            if strength > 0.0:
                inject.append((self.inputs[name], strength))

        fired = self.brain.step(inject=inject)
        rates = self.window.push(fired)
        descending_spikes = int(np.isin(fired, self.descending).sum())
        return fired, rates, descending_spikes
