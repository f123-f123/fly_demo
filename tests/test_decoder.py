import unittest

from app.config import DecoderConfig
from app.decoder import RuleDecoder


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.config = DecoderConfig(
            alpha=1.0,
            deadzone=0.0,
            baseline={name: 0.0 for name in (
                "steer_left", "steer_right", "escape_left", "escape_right"
            )},
            scale={name: 20.0 for name in (
                "steer_left", "steer_right", "escape_left", "escape_right"
            )},
        )
        self.decoder = RuleDecoder(self.config)

    def test_right_activity_produces_positive_turn(self):
        result = self.decoder.decode(
            {
                "steer_left": 0.0,
                "steer_right": 20.0,
                "escape_left": 0.0,
                "escape_right": 0.0,
            },
            elapsed_seconds=0.1,
        )
        self.assertEqual(result.filtered_turn, 1.0)
        self.assertEqual(result.speed, self.config.fixed_speed)

    def test_escape_stops_fixed_speed(self):
        result = self.decoder.decode(
            {
                "steer_left": 0.0,
                "steer_right": 0.0,
                "escape_left": 20.0,
                "escape_right": 0.0,
            },
            elapsed_seconds=0.1,
        )
        self.assertTrue(result.action)
        self.assertEqual(result.speed, 0.0)

    def test_verified_orn_rate_difference_can_bias_turn(self):
        result = self.decoder.decode(
            {
                "steer_left": 0.0,
                "steer_right": 0.0,
                "escape_left": 0.0,
                "escape_right": 0.0,
                "odor_left": 25.5,
                "odor_right": 0.3,
            },
            elapsed_seconds=0.1,
        )
        self.assertLess(result.odor_turn, 0.0)
        self.assertLess(result.filtered_turn, 0.0)


if __name__ == "__main__":
    unittest.main()
