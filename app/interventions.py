from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InterventionStats:
    factor: float
    presynaptic_neurons: int
    affected_connections: int
    original_absolute_weight: float
    scaled_absolute_weight: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class OutgoingWeightScaler:
    """Apply repeatable, in-memory scales to selected presynaptic columns."""

    def __init__(self, brain):
        self.brain = brain
        self.base_weights = np.array(brain.weights, copy=True)
        self.indptr = np.asarray(brain.indptr)
        self.indices = np.asarray(brain.indices)

    def _install(self, weights: np.ndarray) -> None:
        self.brain.weights = weights
        if self.brain.device != "cuda":
            return

        from scipy import sparse
        from cupyx.scipy import sparse as cusparse

        matrix = sparse.csc_matrix(
            (weights, self.indices, self.indptr),
            shape=(self.brain.n, self.brain.n),
        )
        self.brain._W = cusparse.csr_matrix(matrix.tocsr().astype(np.float32))

    def restore(self) -> None:
        self._install(self.base_weights.copy())

    def install(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=self.base_weights.dtype)
        if weights.shape != self.base_weights.shape:
            raise ValueError("weights must match the original sparse data shape")
        self._install(weights.copy())

    def apply(self, presynaptic: np.ndarray, factor: float) -> InterventionStats:
        factor = float(factor)
        if not np.isfinite(factor) or factor < 0.0:
            raise ValueError("factor must be a finite non-negative number")

        neurons = np.unique(np.asarray(presynaptic, dtype=np.int64))
        if len(neurons) == 0:
            raise ValueError("presynaptic population is empty")
        if neurons[0] < 0 or neurons[-1] >= self.brain.n:
            raise IndexError("presynaptic neuron index is outside the brain")

        weights = self.base_weights.copy()
        affected = 0
        original_absolute_weight = 0.0
        for neuron in neurons:
            start = int(self.indptr[neuron])
            stop = int(self.indptr[neuron + 1])
            original_absolute_weight += float(np.abs(self.base_weights[start:stop]).sum())
            weights[start:stop] *= factor
            affected += stop - start

        self._install(weights)
        return InterventionStats(
            factor=factor,
            presynaptic_neurons=len(neurons),
            affected_connections=affected,
            original_absolute_weight=original_absolute_weight,
            scaled_absolute_weight=original_absolute_weight * factor,
        )
