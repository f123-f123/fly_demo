import unittest

import numpy as np

from scripts.run_local_stdp import pair_timing_increment


class PairTimingTests(unittest.TestCase):
    def test_causal_and_anti_causal_order_have_opposite_signs(self):
        edge_pre = np.array([0])
        edge_post = np.array([1])

        before = np.zeros((2, 3), dtype=bool)
        now = np.zeros((2, 3), dtype=bool)
        before[0, 0] = True
        now[0, 1] = True
        causal = pair_timing_increment(before, now, edge_pre, edge_post)

        before.fill(False)
        now.fill(False)
        before[0, 1] = True
        now[0, 0] = True
        anti_causal = pair_timing_increment(before, now, edge_pre, edge_post)

        self.assertGreater(causal[0], 0.0)
        self.assertLess(anti_causal[0], 0.0)

    def test_batch_shift_breaks_within_fly_pairing(self):
        edge_pre = np.array([0])
        edge_post = np.array([1])
        before = np.zeros((2, 3), dtype=bool)
        now = np.zeros((2, 3), dtype=bool)
        before[0, 0] = True
        now[0, 1] = True

        paired = pair_timing_increment(before, now, edge_pre, edge_post)
        shifted = pair_timing_increment(
            before, now, edge_pre, edge_post, post_batch_shift=1
        )

        self.assertGreater(paired[0], 0.0)
        self.assertEqual(shifted[0], 0.0)


if __name__ == "__main__":
    unittest.main()
