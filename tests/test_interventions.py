import unittest

import numpy as np
from scipy import sparse

from app.interventions import OutgoingWeightScaler


class FakeBrain:
    def __init__(self):
        matrix = sparse.csc_matrix(np.array([
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 4.0],
            [0.0, 5.0, 6.0],
        ], dtype=np.float32))
        self.device = "cpu"
        self.n = 3
        self.indptr = matrix.indptr
        self.indices = matrix.indices
        self.weights = matrix.data

    def matrix(self):
        return sparse.csc_matrix(
            (self.weights, self.indices, self.indptr),
            shape=(self.n, self.n),
        ).toarray()


class OutgoingWeightScalerTests(unittest.TestCase):
    def setUp(self):
        self.brain = FakeBrain()
        self.original = self.brain.matrix().copy()
        self.scaler = OutgoingWeightScaler(self.brain)

    def test_scales_only_selected_presynaptic_columns(self):
        stats = self.scaler.apply(np.array([0, 2]), 0.5)

        expected = self.original.copy()
        expected[:, [0, 2]] *= 0.5
        np.testing.assert_allclose(self.brain.matrix(), expected)
        self.assertEqual(stats.presynaptic_neurons, 2)
        self.assertEqual(stats.affected_connections, 4)

    def test_each_condition_starts_from_original_weights(self):
        self.scaler.apply(np.array([0]), 0.0)
        self.scaler.apply(np.array([0]), 2.0)

        expected = self.original.copy()
        expected[:, 0] *= 2.0
        np.testing.assert_allclose(self.brain.matrix(), expected)

    def test_restore_recovers_exact_original_weights(self):
        self.scaler.apply(np.array([1]), 0.5)
        self.scaler.restore()

        np.testing.assert_array_equal(self.brain.matrix(), self.original)

    def test_install_uses_an_explicit_in_memory_copy(self):
        weights = self.scaler.base_weights.copy()
        weights *= 0.25
        self.scaler.install(weights)
        weights[:] = 0.0

        np.testing.assert_allclose(self.brain.matrix(), self.original * 0.25)
        with self.assertRaises(ValueError):
            self.scaler.install(np.zeros(1, dtype=np.float32))

    def test_rejects_invalid_interventions(self):
        with self.assertRaises(ValueError):
            self.scaler.apply(np.array([], dtype=np.int64), 1.0)
        with self.assertRaises(ValueError):
            self.scaler.apply(np.array([0]), -1.0)
        with self.assertRaises(IndexError):
            self.scaler.apply(np.array([3]), 1.0)


if __name__ == "__main__":
    unittest.main()
