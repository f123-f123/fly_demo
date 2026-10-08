import unittest

import numpy as np

from app.activity import ActivityWindow


class ActivityWindowTests(unittest.TestCase):
    def test_rate_uses_rolling_window_and_population_size(self):
        window = ActivityWindow(
            {"group": np.array([1, 2], dtype=np.int64)},
            dt=0.02,
            window_steps=3,
        )
        window.push(np.array([1]))
        window.push(np.array([2]))
        rates = window.push(np.array([], dtype=np.int64))
        self.assertAlmostEqual(rates["group"], 2 / 2 / 0.06)

    def test_empty_population_is_rejected(self):
        with self.assertRaises(ValueError):
            ActivityWindow({"empty": np.array([], dtype=np.int64)}, 0.02, 3)


if __name__ == "__main__":
    unittest.main()

