import math
import unittest

import numpy as np

from app.encoder import FeatureEncoder
from app.world import Collider, Sensors, World


class WorldEncoderTests(unittest.TestCase):
    def test_target_side_changes_lc10a_balance(self):
        world = World()
        encoder = FeatureEncoder()
        world.set_target(6.0, 4.0)
        right_stimulus = encoder.encode(world.sense(0.02))
        self.assertGreater(right_stimulus.lc10a_right, right_stimulus.lc10a_left)

        world.set_target(6.0, -4.0)
        left_stimulus = encoder.encode(world.sense(0.02))
        self.assertGreater(left_stimulus.lc10a_left, left_stimulus.lc10a_right)

    def test_fixed_speed_moves_fly_in_heading_direction(self):
        world = World()
        world.fly.heading = math.pi / 2
        for _ in range(50):
            world.update(0.02, speed=2.0, turn=0.0, max_turn_rate=1.0)
        self.assertLess(abs(world.fly.x), 0.25)
        self.assertGreater(world.fly.z, 0.8)

    def test_approaching_obstacle_generates_looming(self):
        world = World()
        world.spawn_obstacle(side=0.0, distance=5.0)
        world.sense(0.02)
        world.fly.x = 1.0
        sensors = world.sense(0.02)
        self.assertGreater(sensors.looming_rate, 0.0)

        stimulus = FeatureEncoder().encode(sensors)
        self.assertGreater(stimulus.loom_left, 0.0)
        self.assertAlmostEqual(stimulus.loom_left, stimulus.loom_right)

    def test_height_hold_policy_produces_real_vertical_motion(self):
        world = World(seed=4)
        initial_y = world.fly.y
        for _ in range(50):
            world.update(0.02, speed=0.0, turn=0.0, max_turn_rate=1.0)
        self.assertGreater(world.fly.y, initial_y)
        self.assertGreater(world.fly.vy, 0.0)

    def test_visual_occlusion_is_reported(self):
        world = World(seed=2)
        for source in world.food_sources:
            source.enabled = False
        world.set_target(5.0, 0.0, y=0.65)
        world.colliders = [Collider("test", "rock", 2.5, 0.65, 0.0, 0.7)]
        sensors = world.sense(0.02)
        self.assertFalse(sensors.target_visible)
        self.assertTrue(sensors.target_occluded)

    def test_bilateral_antennae_sample_a_spatial_gradient(self):
        world = World(seed=3)
        for source in world.food_sources:
            source.enabled = False
        source = world.food_sources[0]
        source.enabled = True
        source.x, source.y, source.z = 3.0, world.fly.y, -1.0
        source.odor_emission = 0.3
        world.wind = (1.0, 0.0, 0.0)
        sensors = world.sense(0.02)
        self.assertGreater(sensors.odor_left, sensors.odor_right)
        stimulus = FeatureEncoder().encode(sensors)
        self.assertGreater(stimulus.odor_left, stimulus.odor_right)

    def test_world_statistics_include_completed_trial_rates(self):
        world = World(seed=5)
        world.stats["sources_reached"] = 2
        world.stats["landings"] = 1
        world.stats["threats_avoided"] = 3
        world.stats["threat_contacts"] = 1
        statistics = world.statistics()
        self.assertEqual(statistics["threat_trials"], 4)
        self.assertAlmostEqual(statistics["avoidance_success_rate"], 0.75)
        self.assertAlmostEqual(statistics["landing_success_rate"], 0.5)

    def test_reset_generates_a_new_reproducible_layout(self):
        world = World(seed=8)
        first = [(source.x, source.z) for source in world.food_sources]
        world.reset()
        second = [(source.x, source.z) for source in world.food_sources]
        twin = World(seed=8)
        twin.reset()
        self.assertNotEqual(first, second)
        self.assertEqual(second, [(source.x, source.z) for source in twin.food_sources])

    def test_exploration_height_is_not_locked_to_a_constant(self):
        world = World(seed=9)
        for source in world.food_sources:
            source.enabled = False
        world._next_threat_time = math.inf
        heights = []
        for _ in range(900):
            sensors = world.sense(0.02)
            world.update(0.02, speed=0.0, turn=0.0, max_turn_rate=1.0, sensors=sensors)
            heights.append(world.fly.y)
        self.assertGreater(max(heights) - min(heights), 0.45)
        self.assertEqual(world.fly.behavior_state, "explore")

    def test_looming_cannot_bypass_neural_action(self):
        world = World(seed=10)
        sensors = world.sense(0.02)
        sensors.looming_rate = 0.2
        world.update(0.02, speed=1.0, turn=0.0, max_turn_rate=1.0, sensors=sensors)
        self.assertNotEqual(world.fly.behavior_state, "escape")
        self.assertGreater(world.motion_policy["effective_speed"], 0.0)

    def test_neural_escape_uses_explicit_direction_policy(self):
        for elevation, direction in ((0.4, -1), (-0.4, 1)):
            world = World(seed=10)
            world.colliders = []
            world.fly.y = 2.5
            world.fly.flight_mode = "hovering"
            sensors = Sensors(obstacle_visible=True, obstacle_elevation=elevation)
            world.update(.02, 1., 0., 1., action=True, sensors=sensors)
            self.assertEqual(world.fly.behavior_state, "escape")
            self.assertEqual(world.motion_policy["escape_trigger"], "DNp01")
            self.assertEqual(world.motion_policy["escape_direction_policy"], direction)
            self.assertGreater(world.fly.vy * direction, 0.)

    def test_manual_action_without_threat_has_no_vertical_kick(self):
        world = World()
        world.colliders = []
        world.fly.y = 2.5
        world.fly.flight_mode = "hovering"
        world.update(.02, 1., 0., 1., action=True, sensors=Sensors())
        self.assertEqual(world.motion_policy["height_policy"], "DNp01_brake_hold")
        self.assertLess(abs(world.fly.vy), .05)

    def test_escape_uses_available_clearance_near_floor_and_ceiling(self):
        for height, elevation, direction in ((1.0,.4,1), (5.0,-.4,-1)):
            world = World()
            world.fly.y = height
            world._select_escape_height(Sensors(obstacle_visible=True, obstacle_elevation=elevation))
            self.assertEqual(world._escape_direction, direction)

    def test_escape_policy_clears_a_visible_incoming_threat(self):
        world = World(seed=10)
        world.colliders = []
        world._next_threat_time = math.inf
        world.fly.y = 2.5
        world.fly.flight_mode = "hovering"
        world.spawn_obstacle(side=0.0, distance=7.0)
        for step in range(180):
            sensors = world.sense(0.02)
            world.update(
                0.02,
                speed=0.0,
                turn=0.0,
                max_turn_rate=1.0,
                action=40 <= step < 80,
                sensors=sensors,
            )
            if not world.obstacle.enabled:
                break
        self.assertEqual(world.stats["threat_contacts"], 0)
        self.assertEqual(world.stats["threats_avoided"], 1)

    def test_distant_visual_lock_does_not_pin_flight_to_food_height(self):
        world = World()
        world.fly.flight_mode = "hovering"
        world.fly.behavior_state = "visual_lock"
        desired = world._desired_height(Sensors(target_visible=True,target_distance=10.,target_elevation=-.5))
        self.assertAlmostEqual(desired, world._exploration_height)

    def test_exploration_height_changes_continuously(self):
        world = World()
        previous = world._exploration_height
        for _ in range(1000):
            world._update_microdynamics(.02)
            self.assertLess(abs(world._exploration_height-previous), .03)
            previous = world._exploration_height

    def test_odor_rate_does_not_imply_vertical_direction(self):
        world = World()
        world.fly.flight_mode = "hovering"
        world.fly.behavior_state = "odor_tracking"
        self.assertEqual(world._desired_height(Sensors(odor_rate=10.)),
                         world._desired_height(Sensors(odor_rate=-10.)))

    def test_visual_height_corrects_for_body_pitch_and_bearing(self):
        world = World()
        world.fly.x, world.fly.y, world.fly.z = 0., 2., 0.
        world.fly.pitch = .4
        world.fly.behavior_state = "approach"
        world.fly.flight_mode = "hovering"
        distance, bearing, elevation = world._body_angles(3., 1.5, 2.)
        sensors = Sensors(target_visible=True, target_distance=distance, target_bearing=bearing,
                          target_elevation=elevation, target_angular_size=2*math.atan(.3/distance))
        self.assertAlmostEqual(world._desired_height(sensors), 1.5+.3+world.fly.radius)

    def test_branch_collision_and_occlusion_use_thin_capsule(self):
        world = World()
        world.colliders = [Collider("twig", "branch", 0., 2., -1., .05,
                                    shape="capsule", end=(0., 2., 1.))]
        world.fly.x, world.fly.y, world.fly.z = .6, 2., 0.
        world._resolve_collisions()
        self.assertEqual(world.stats["collisions"], 0)
        twig = world.colliders[0]
        self.assertFalse(twig.blocks_segment(np.array([-1., 2.5, 0.]), np.array([1., 2.5, 0.])))
        self.assertTrue(twig.blocks_segment(np.array([-1., 2., 0.]), np.array([1., 2., 0.])))
        world.fly.x = .1
        world.fly.vx = -1.
        world._resolve_collisions()
        self.assertAlmostEqual(world.fly.x, world.fly.radius+.05)
        self.assertEqual(world.fly.vy, 0.)
        self.assertEqual(world.fly.vx, 0.)

    def test_leaf_surface_collision_and_occlusion(self):
        world = World()
        leaf = Collider("leaf", "foliage", 0., 2., 0., .008, shape="leaf",
                        vertices=((-1.,2.,-1.), (1.,2.,-1.), (0.,2.,1.)))
        world.colliders = [leaf]
        self.assertTrue(leaf.blocks_segment(np.array([0.,1.,0.]),np.array([0.,3.,0.])))
        self.assertFalse(leaf.blocks_segment(np.array([.9,1.,.9]),np.array([.9,3.,.9])))
        world.fly.x, world.fly.y, world.fly.z = 0., 1.9, 0.
        world.fly.vy = 1.
        world._resolve_collisions()
        self.assertLess(world.fly.y, 1.9)
        self.assertEqual(world.fly.vy, 0.)

    def test_plants_are_grounded_and_reset_geometry_is_reproducible(self):
        world = World(seed=18)
        for collider in world.colliders:
            if collider.id.endswith("-stem"):
                self.assertAlmostEqual(collider.y,world.ground_height_at(collider.x,collider.z))
        world.reset()
        twin = World(seed=18)
        twin.reset()
        self.assertEqual(world.colliders, twin.colliders)

    def test_ground_height_varies_across_the_randomized_terrain(self):
        world = World(seed=11)
        heights = {round(world.ground_height_at(x, z), 4) for x, z in ((-8, -8), (0, 0), (8, 8), (-8, 8))}
        self.assertGreater(len(heights), 2)

    def test_ceiling_recovery_moves_away_without_repeated_collisions(self):
        world = World(seed=12)
        world.colliders = []
        world._next_threat_time = math.inf
        world.fly.flight_mode = "hovering"
        world.fly.y = world.glass_height - world.fly.radius
        world.fly.vy = 1.0
        for _ in range(50):
            sensors = world.sense(0.02)
            world.update(0.02, speed=0.0, turn=0.0, max_turn_rate=1.0, sensors=sensors)
        self.assertLessEqual(world.stats["collisions"], 1)
        self.assertLess(world.fly.y, world.glass_height - world.fly.radius)


if __name__ == "__main__":
    unittest.main()
