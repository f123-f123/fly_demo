from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Fly:
    x: float = 0.0
    y: float = 0.18
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    radius: float = 0.22
    flight_phase: float = 0.0
    flight_mode: str = "taking_off"
    behavior_state: str = "explore"

    @property
    def heading(self) -> float:
        return self.yaw

    @heading.setter
    def heading(self, value: float) -> None:
        self.yaw = value

    @property
    def altitude(self) -> float:
        return self.y


@dataclass
class Target:
    x: float = 7.0
    y: float = 0.65
    z: float = 3.0
    radius: float = 0.24
    contrast: float = 1.0
    enabled: bool = False
    orbit: bool = False
    orbit_phase: float = 0.0
    kind: str = "diagnostic_target"


@dataclass
class FoodSource:
    id: str
    kind: str
    x: float
    y: float
    z: float
    radius: float
    odor_emission: float
    contrast: float
    enabled: bool = True
    moving: bool = False
    phase: float = 0.0


@dataclass
class Collider:
    id: str
    kind: str
    x: float
    y: float
    z: float
    radius: float
    # Coordinates are world-space and shared verbatim with the renderer.
    shape: str = "sphere"
    end: tuple[float, float, float] | None = None
    vertices: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        points = self.vertices or ((self.x, self.y, self.z), self.end or (self.x, self.y, self.z))
        self.bounds = tuple((min(p[axis] for p in points) - self.radius,
                             max(p[axis] for p in points) + self.radius) for axis in range(3))

    def near_point(self, point: np.ndarray, padding: float) -> bool:
        return all(low - padding <= point[axis] <= high + padding
                   for axis, (low, high) in enumerate(self.bounds))

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        start = np.array([self.x, self.y, self.z])
        if self.shape == "capsule":
            return closest_on_segment(point, start, np.asarray(self.end))
        if self.shape == "leaf":
            a, b, c = map(np.asarray, self.vertices)
            normal = np.cross(b - a, c - a)
            normal /= max(float(np.linalg.norm(normal)), 1e-9)
            projected = point - normal * float((point - a) @ normal)
            if all(float(np.cross(v - u, projected - u) @ normal) >= -1e-9
                   for u, v in ((a, b), (b, c), (c, a))):
                return projected
            candidates = [closest_on_segment(point, u, v) for u, v in ((a, b), (b, c), (c, a))]
            return min(candidates, key=lambda p: float(np.sum((point - p) ** 2)))
        return start

    def blocks_segment(self, origin: np.ndarray, endpoint: np.ndarray) -> bool:
        entry, leave = 0.0, 1.0
        for axis, (low, high) in enumerate(self.bounds):
            delta = float(endpoint[axis] - origin[axis])
            if abs(delta) < 1e-12:
                if not low <= origin[axis] <= high:
                    return False
                continue
            first, last = (low - origin[axis]) / delta, (high - origin[axis]) / delta
            entry, leave = max(entry, min(first, last)), min(leave, max(first, last))
            if entry > leave:
                return False
        start = np.array([self.x, self.y, self.z])
        if self.shape == "leaf":
            a, b, c = map(np.asarray, self.vertices)
            normal = np.cross(b - a, c - a)
            denominator = float((endpoint - origin) @ normal)
            if abs(denominator) < 1e-9:
                return False
            fraction = float((a - origin) @ normal) / denominator
            if not 0.001 < fraction < 0.999:
                return False
            point = origin + fraction * (endpoint - origin)
            return float(np.linalg.norm(point - self.closest_point(point))) < 1e-6
        end = np.asarray(self.end) if self.shape == "capsule" else start
        return segment_distance(origin, endpoint, start, end) < self.radius


def closest_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = b - a
    fraction = float(np.clip(float((point - a) @ direction) / max(float(direction @ direction), 1e-12), 0.0, 1.0))
    return a + fraction * direction


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    # The constrained minimum is either an interior stationary point or on an edge.
    u, v, w = b - a, d - c, a - c
    uu, vv, uv = float(u @ u), float(v @ v), float(u @ v)
    candidates = [np.linalg.norm(p - closest_on_segment(p, x, y))
                  for p, x, y in ((a, c, d), (b, c, d), (c, a, b), (d, a, b))]
    determinant = uu * vv - uv * uv
    if determinant > 1e-12:
        s = (uv * float(v @ w) - vv * float(u @ w)) / determinant
        t = (uu * float(v @ w) - uv * float(u @ w)) / determinant
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            candidates.append(np.linalg.norm(a + s * u - c - t * v))
    return float(min(candidates))


@dataclass
class WindZone:
    id: str
    x: float
    y: float
    z: float
    radius: float
    vx: float
    vy: float
    vz: float


@dataclass
class Obstacle:
    x: float = 0.0
    y: float = 1.2
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    radius: float = 0.7
    enabled: bool = False
    previous_angular_size: float = 0.0
    approach_speed: float = 3.2
    age: float = 0.0


@dataclass
class Sensors:
    target_visible: bool = False
    target_id: str = "none"
    target_kind: str = "none"
    target_distance: float = 0.0
    target_bearing: float = 0.0
    target_elevation: float = 0.0
    target_bearing_normalized: float = 0.0
    target_elevation_normalized: float = 0.0
    target_angular_size: float = 0.0
    target_angular_rate: float = 0.0
    target_contrast: float = 0.0
    target_occluded: bool = False
    target_closing_speed: float = 0.0
    target_ttc: float = math.inf
    obstacle_visible: bool = False
    obstacle_distance: float = 0.0
    obstacle_bearing: float = 0.0
    obstacle_elevation: float = 0.0
    angular_size: float = 0.0
    looming_rate: float = 0.0
    obstacle_closing_speed: float = 0.0
    obstacle_ttc: float = math.inf
    odor_left: float = 0.0
    odor_right: float = 0.0
    odor_mean: float = 0.0
    odor_difference: float = 0.0
    odor_rate: float = 0.0


class World:
    half_extent = 15.0
    glass_height = 5.5
    half_horizontal_fov = math.radians(135.0)
    half_vertical_fov = math.radians(52.0)

    def __init__(self, half_fov_degrees: float = 135.0, seed: int = 1):
        self.half_horizontal_fov = math.radians(half_fov_degrees)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fly = Fly()
        self.target = Target()
        self.obstacle = Obstacle()
        self.food_sources: list[FoodSource] = []
        self.colliders: list[Collider] = []
        self.wind_zones: list[WindZone] = []
        self.wind = (0.55, 0.0, 0.22)
        self.layout_index = 0
        self.terrain = {"phase_x": 0.0, "phase_z": 0.0, "ridge": 0.0}
        self.time = 0.0
        self.stats = {
            "fruits_found": 0,
            "sources_reached": 0,
            "landings": 0,
            "collisions": 0,
            "threats_avoided": 0,
            "threat_contacts": 0,
        }
        self._previous_visual_sizes: dict[str, float] = {}
        self._previous_odor_mean = 0.0
        self._mode_time = 0.0
        self._behavior_time = 0.0
        self._was_action = False
        self._threat_contact = False
        self._next_threat_time = float(self.rng.uniform(6.0, 10.0))
        self._found_ids: set[str] = set()
        self._perched_source_id: str | None = None
        self._vertical_noise = 0.0
        self._pitch_noise = 0.0
        self._yaw_drift = 0.0
        self._gust = np.zeros(3, dtype=float)
        self._exploration_height = 1.35
        self._escape_height = 1.35
        self._height_policy = "exploration"
        self._escape_direction = 0.0
        self._saccade_remaining = 0.0
        self._saccade_cooldown = 0.0
        self._saccade_sign = 0.0
        self._collision_recovery = 0.0
        self._colliding_ids: set[str] = set()
        self.motion_policy: dict[str, float | str | bool] = {}
        self._randomize_layout()
        self.fly.y = self.ground_height_at(0.0, 0.0) + self.fly.radius

    def statistics(self) -> dict[str, float | int]:
        """Return counters plus rates derived only from completed world events."""
        threat_trials = self.stats["threats_avoided"] + self.stats["threat_contacts"]
        source_trials = self.stats["sources_reached"]
        return {
            **self.stats,
            "threat_trials": threat_trials,
            "avoidance_success_rate": (
                self.stats["threats_avoided"] / threat_trials if threat_trials else 0.0
            ),
            "landing_success_rate": (
                self.stats["landings"] / source_trials if source_trials else 0.0
            ),
        }

    @property
    def car(self) -> Fly:
        """Compatibility alias retained for earlier tests and local scripts."""
        return self.fly

    def ground_height_at(self, x: float, z: float) -> float:
        height = (
            0.08
            + 0.075 * math.sin(x * 0.34 + self.terrain["phase_x"])
            + 0.055 * math.sin(z * 0.29 + self.terrain["phase_z"])
            + 0.035 * math.sin((x + z) * 0.71 + self.terrain["ridge"])
        )
        return float(np.clip(height, 0.015, 0.24))

    def _randomize_layout(self) -> None:
        self.terrain = {
            "phase_x": float(self.rng.uniform(-math.pi, math.pi)),
            "phase_z": float(self.rng.uniform(-math.pi, math.pi)),
            "ridge": float(self.rng.uniform(-math.pi, math.pi)),
        }
        wind_angle = float(self.rng.uniform(-0.55, 0.55))
        self.wind = (0.55 * math.cos(wind_angle), 0.0, 0.55 * math.sin(wind_angle))
        self._generate_ecology()

    def _cluster_center(self, minimum_distance: float = 5.0) -> tuple[float, float]:
        angle = float(self.rng.uniform(-math.pi, math.pi))
        distance = float(self.rng.uniform(minimum_distance, 10.5))
        return math.cos(angle) * distance, math.sin(angle) * distance

    def _generate_ecology(self) -> None:
        self.food_sources = []
        fruit_center = self._cluster_center()
        flower_center = self._cluster_center()
        while math.hypot(flower_center[0] - fruit_center[0], flower_center[1] - fruit_center[1]) < 5.0:
            flower_center = self._cluster_center()
        kinds = ("fruit", "fruit", "fruit", "fruit", "flower", "flower", "flower")
        for index, kind in enumerate(kinds):
            is_fruit = kind == "fruit"
            center = fruit_center if is_fruit else flower_center
            x = float(np.clip(center[0] + self.rng.normal(0.0, 1.25), -12.5, 12.5))
            z = float(np.clip(center[1] + self.rng.normal(0.0, 1.25), -12.5, 12.5))
            ground = self.ground_height_at(x, z)
            self.food_sources.append(FoodSource(
                id=f"{kind}-{index + 1}",
                kind=kind,
                x=x,
                y=float(ground + 0.34 if is_fruit else ground + self.rng.uniform(0.7, 1.45)),
                z=z,
                radius=0.34 if is_fruit else 0.26,
                odor_emission=float(self.rng.uniform(0.9, 1.15) if is_fruit else self.rng.uniform(0.45, 0.68)),
                contrast=0.95 if is_fruit else 0.78,
                moving=False,
                phase=float(self.rng.uniform(-math.pi, math.pi)),
            ))

        self.colliders = []
        for kind, count in (("rock", 7), ("foliage", 9), ("branch", 4)):
            for index in range(count):
                while True:
                    x, z = self.rng.uniform(-12.8, 12.8, size=2)
                    if math.hypot(float(x), float(z)) > 2.2:
                        break
                radius = float(self.rng.uniform(0.55, 0.95) if kind == "rock" else self.rng.uniform(0.65, 1.15))
                ground = self.ground_height_at(float(x), float(z))
                name = f"{kind}-{index + 1}"
                if kind == "rock":
                    self.colliders.append(Collider(name, kind, float(x), ground + radius * 0.55, float(z), radius))
                    continue
                base = np.array([x, ground, z], dtype=float)
                height = float(self.rng.uniform(1.4, 3.9))
                angle = float(self.rng.uniform(-math.pi, math.pi))
                lean = np.array([math.cos(angle) * 0.45, height, math.sin(angle) * 0.45])
                top = base + lean
                thickness = 0.10 if kind == "branch" else 0.045
                self.colliders.append(Collider(name + "-stem", "branch", *map(float, base), thickness,
                                               shape="capsule", end=tuple(map(float, top))))
                for tier in range(5):
                    start = base + lean * (0.28 + tier * 0.14)
                    azimuth = angle + tier * 2.4
                    radial = np.array([math.cos(azimuth), 0.25, math.sin(azimuth)])
                    tip = start + radial * radius * (1.1 - tier * 0.1)
                    self.colliders.append(Collider(f"{name}-twig-{tier}", "branch", *map(float, start), thickness * 0.45,
                                                   shape="capsule", end=tuple(map(float, tip))))
                    if kind == "branch" and tier < 2:
                        continue
                    # A pointed, folded leaf: two thin triangles, not a canopy sphere.
                    side = np.array([-math.sin(azimuth), 0.0, math.cos(azimuth)]) * radius * 0.24
                    mid = tip + radial * radius * 0.35
                    leaf_tip = tip + radial * radius * 0.85 + np.array([0.0, -0.18, 0.0])
                    for half, edge in enumerate((mid + side, mid - side)):
                        points = (tip, edge, leaf_tip) if half == 0 else (tip, leaf_tip, edge)
                        center = sum(points) / 3.0
                        self.colliders.append(Collider(f"{name}-leaf-{tier}-{half}", "foliage", *map(float, center), 0.008,
                                                       shape="leaf", vertices=tuple(tuple(map(float, p)) for p in points)))

        self.wind_zones = []
        for index in range(3):
            x, z = self.rng.uniform(-10.0, 10.0, size=2)
            angle = float(self.rng.uniform(-math.pi, math.pi))
            strength = float(self.rng.uniform(0.18, 0.42))
            self.wind_zones.append(WindZone(
                id=f"wind-{index + 1}", x=float(x), y=float(self.rng.uniform(0.8, 3.2)), z=float(z),
                radius=float(self.rng.uniform(3.0, 5.5)), vx=math.cos(angle) * strength,
                vy=float(self.rng.uniform(-0.08, 0.12)), vz=math.sin(angle) * strength,
            ))

    def reset(self) -> None:
        self.layout_index += 1
        self.rng = np.random.default_rng(self.seed + self.layout_index * 9973)
        self.fly = Fly()
        self.target = Target()
        self.obstacle = Obstacle()
        self.time = 0.0
        self.stats = {name: 0 for name in self.stats}
        self._previous_visual_sizes.clear()
        self._previous_odor_mean = 0.0
        self._mode_time = 0.0
        self._behavior_time = 0.0
        self._was_action = False
        self._threat_contact = False
        self._next_threat_time = float(self.rng.uniform(6.0, 10.0))
        self._found_ids.clear()
        self._perched_source_id = None
        self._vertical_noise = 0.0
        self._pitch_noise = 0.0
        self._yaw_drift = 0.0
        self._gust = np.zeros(3, dtype=float)
        self._exploration_height = 1.35
        self._escape_height = 1.35
        self._height_policy = "exploration"
        self._escape_direction = 0.0
        self._saccade_remaining = 0.0
        self._saccade_cooldown = 0.0
        self._saccade_sign = 0.0
        self._collision_recovery = 0.0
        self._colliding_ids.clear()
        self.motion_policy = {}
        self._randomize_layout()
        self.fly.y = self.ground_height_at(0.0, 0.0) + self.fly.radius

    def set_target(self, x: float, z: float, y: float = 0.65) -> None:
        self.target.x = float(np.clip(x, -14.0, 14.0))
        self.target.y = float(np.clip(y, 0.25, self.glass_height - 0.25))
        self.target.z = float(np.clip(z, -14.0, 14.0))
        self.target.enabled = True

    def spawn_obstacle(self, side: float = 0.0, distance: float = 7.0) -> None:
        angle = self.fly.yaw + side * math.radians(28.0)
        elevation = math.radians(8.0)
        self.obstacle.x = self.fly.x + math.cos(angle) * math.cos(elevation) * distance
        self.obstacle.y = min(self.glass_height - 0.8, self.fly.y + math.sin(elevation) * distance)
        self.obstacle.z = self.fly.z + math.sin(angle) * math.cos(elevation) * distance
        dx = self.fly.x - self.obstacle.x
        dy = self.fly.y - self.obstacle.y
        dz = self.fly.z - self.obstacle.z
        length = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)
        speed = self.obstacle.approach_speed
        self.obstacle.vx = dx / length * speed
        self.obstacle.vy = dy / length * speed
        self.obstacle.vz = dz / length * speed
        self.obstacle.enabled = True
        self.obstacle.previous_angular_size = 0.0
        self.obstacle.age = 0.0
        self._threat_contact = False

    def _body_angles(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        dx, dy, dz = x - self.fly.x, y - self.fly.y, z - self.fly.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        horizontal_forward = math.cos(self.fly.yaw) * dx + math.sin(self.fly.yaw) * dz
        lateral = -math.sin(self.fly.yaw) * dx + math.cos(self.fly.yaw) * dz
        forward = math.cos(self.fly.pitch) * horizontal_forward + math.sin(self.fly.pitch) * dy
        vertical = -math.sin(self.fly.pitch) * horizontal_forward + math.cos(self.fly.pitch) * dy
        azimuth = wrap_angle(math.atan2(lateral, forward))
        elevation = math.atan2(vertical, max(math.hypot(forward, lateral), 1e-6))
        return distance, azimuth, elevation

    def _occluded(self, x: float, y: float, z: float, target_radius: float) -> bool:
        origin = np.array([self.fly.x, self.fly.y, self.fly.z], dtype=float)
        endpoint = np.array([x, y, z], dtype=float)
        segment = endpoint - origin
        length_squared = float(segment @ segment)
        if length_squared <= 1e-9:
            return False
        # Vectorized slab broad phase keeps richer plant geometry within 20 ms.
        if getattr(self, "_bounds_owner", None) is not self.colliders or getattr(self, "_bounds_count", -1) != len(self.colliders):
            self._bounds_owner = self.colliders
            self._bounds_count = len(self.colliders)
            self._bounds_array = np.array([c.bounds for c in self.colliders], dtype=float).reshape(-1, 3, 2)
        bounds = self._bounds_array
        if len(bounds) == 0:
            return False
        parallel = np.abs(segment) < 1e-12
        safe_direction = np.where(parallel, 1.0, segment)
        first = (bounds[:, :, 0] - origin) / safe_direction
        last = (bounds[:, :, 1] - origin) / safe_direction
        low, high = np.minimum(first, last), np.maximum(first, last)
        low[:, parallel], high[:, parallel] = -math.inf, math.inf
        outside = np.any(parallel & ((origin < bounds[:, :, 0]) | (origin > bounds[:, :, 1])), axis=1)
        candidates = ~outside & (np.maximum(low.max(axis=1), 0.0) <= np.minimum(high.min(axis=1), 1.0))
        for index in np.flatnonzero(candidates):
            collider = self.colliders[index]
            if collider.blocks_segment(origin, endpoint):
                return True
        return False

    def _closing_speed(self, x: float, y: float, z: float, vx: float, vy: float, vz: float) -> float:
        relative = np.array([x - self.fly.x, y - self.fly.y, z - self.fly.z], dtype=float)
        distance = max(float(np.linalg.norm(relative)), 1e-6)
        relative_velocity = np.array([vx - self.fly.vx, vy - self.fly.vy, vz - self.fly.vz], dtype=float)
        return max(0.0, -float(relative @ relative_velocity) / distance)

    def _visual_candidate(self, dt: float) -> tuple[dict[str, float | str | bool], float]:
        candidates: list[tuple[str, str, float, float, float, float, float, float, float, float]] = []
        if self.target.enabled:
            candidates.append((self.target.kind, "manual-target", self.target.x, self.target.y, self.target.z, self.target.radius, self.target.contrast, 0.0, 0.0, 0.0))
        for source in self.food_sources:
            if source.enabled:
                candidates.append((source.kind, source.id, source.x, source.y, source.z, source.radius, source.contrast, 0.0, 0.0, 0.0))

        best: dict[str, float | str | bool] | None = None
        nearest: dict[str, float | str | bool] | None = None
        best_score = -1.0
        nearest_distance = math.inf
        for kind, object_id, x, y, z, radius, contrast, vx, vy, vz in candidates:
            distance, azimuth, elevation = self._body_angles(x, y, z)
            nearest_distance = min(nearest_distance, distance)
            angular_size = 2.0 * math.atan2(radius, max(distance, 0.05))
            previous = self._previous_visual_sizes.get(object_id, angular_size)
            angular_rate = (angular_size - previous) / dt
            self._previous_visual_sizes[object_id] = angular_size
            occluded = self._occluded(x, y, z, radius)
            visible = (
                distance <= 20.0
                and abs(azimuth) <= self.half_horizontal_fov
                and abs(elevation) <= self.half_vertical_fov
                and not occluded
            )
            closing = self._closing_speed(x, y, z, vx, vy, vz)
            record: dict[str, float | str | bool] = {
                "visible": visible,
                "id": object_id,
                "kind": kind,
                "distance": distance,
                "azimuth": azimuth,
                "elevation": elevation,
                "angular_size": angular_size,
                "angular_rate": angular_rate,
                "contrast": contrast,
                "occluded": occluded,
                "closing_speed": closing,
                "ttc": max(distance - radius, 0.0) / closing if closing > 1e-5 else math.inf,
            }
            if nearest is None or distance < float(nearest["distance"]):
                nearest = record
            score = contrast * angular_size * max(0.0, math.cos(azimuth)) if visible else -1.0
            if object_id == "manual-target" and visible:
                score *= 3.0
            if score <= best_score:
                continue
            best_score = score
            best = record
        return best or nearest or {
            "visible": False, "id": "none", "kind": "none", "distance": nearest_distance if math.isfinite(nearest_distance) else 0.0,
            "azimuth": 0.0, "elevation": 0.0, "angular_size": 0.0, "angular_rate": 0.0,
            "contrast": 0.0, "occluded": False, "closing_speed": 0.0, "ttc": math.inf,
        }, nearest_distance

    def _wind_at(self, x: float, y: float, z: float) -> np.ndarray:
        local = np.array(self.wind, dtype=float)
        for zone in self.wind_zones:
            distance_squared = (
                (x - zone.x) ** 2 + (y - zone.y) ** 2 + (z - zone.z) ** 2
            )
            weight = math.exp(-distance_squared / max(2.0 * zone.radius * zone.radius, 1e-6))
            local += np.array([zone.vx, zone.vy, zone.vz], dtype=float) * weight
        return local

    def _odor_at(self, x: float, y: float, z: float) -> float:
        wind = self._wind_at(x, y, z)
        wind_norm = max(float(np.linalg.norm(wind)), 1e-6)
        wind_direction = wind / wind_norm
        sample = np.array([x, y, z], dtype=float)
        concentration = 0.0
        for source in self.food_sources:
            if not source.enabled:
                continue
            relative = sample - np.array([source.x, source.y, source.z], dtype=float)
            distance = max(float(np.linalg.norm(relative)), 0.08)
            downwind = float(relative @ wind_direction)
            crosswind = float(np.linalg.norm(relative - wind_direction * downwind))
            plume_width = 0.55 + 0.22 * max(downwind, 0.0)
            plume = math.exp(-(crosswind * crosswind) / (2.0 * plume_width * plume_width))
            if downwind < 0.0:
                plume *= 0.08
            decay = math.exp(-distance / 11.0) / (1.0 + 0.075 * distance * distance)
            noise = 1.0 + 0.09 * math.sin(self.time * 5.3 + source.phase + x * 0.8 + z * 0.35)
            concentration += source.odor_emission * decay * (0.12 + 0.88 * plume) * noise
        return max(0.0, concentration)

    def _odor_sensors(self, dt: float) -> tuple[float, float, float, float, float]:
        forward_x, forward_z = math.cos(self.fly.yaw), math.sin(self.fly.yaw)
        right_x, right_z = -math.sin(self.fly.yaw), math.cos(self.fly.yaw)
        head_x = self.fly.x + forward_x * 0.2
        head_y = self.fly.y + 0.04
        head_z = self.fly.z + forward_z * 0.2
        left = self._odor_at(head_x - right_x * 0.13, head_y, head_z - right_z * 0.13)
        right = self._odor_at(head_x + right_x * 0.13, head_y, head_z + right_z * 0.13)
        mean = (left + right) * 0.5
        rate = (mean - self._previous_odor_mean) / dt
        self._previous_odor_mean = mean
        return left, right, mean, right - left, rate

    def sense(self, dt: float) -> Sensors:
        visual, _ = self._visual_candidate(dt)
        obstacle_distance, obstacle_bearing, obstacle_elevation = self._body_angles(
            self.obstacle.x, self.obstacle.y, self.obstacle.z
        )
        obstacle_occluded = self._occluded(self.obstacle.x, self.obstacle.y, self.obstacle.z, self.obstacle.radius)
        obstacle_visible = (
            self.obstacle.enabled
            and obstacle_distance <= 16.0
            and abs(obstacle_bearing) <= self.half_horizontal_fov
            and abs(obstacle_elevation) <= self.half_vertical_fov
            and not obstacle_occluded
        )
        angular_size = 0.0
        looming_rate = 0.0
        closing_speed = 0.0
        obstacle_ttc = math.inf
        if obstacle_visible:
            angular_size = 2.0 * math.atan2(self.obstacle.radius, max(obstacle_distance, 0.05))
            if self.obstacle.previous_angular_size > 0.0:
                looming_rate = max(0.0, (angular_size - self.obstacle.previous_angular_size) / dt)
            self.obstacle.previous_angular_size = angular_size
            closing_speed = self._closing_speed(
                self.obstacle.x, self.obstacle.y, self.obstacle.z,
                self.obstacle.vx, self.obstacle.vy, self.obstacle.vz,
            )
            if closing_speed > 1e-5:
                obstacle_ttc = max(obstacle_distance - self.obstacle.radius, 0.0) / closing_speed
        elif self.obstacle.enabled:
            self.obstacle.previous_angular_size = 0.0

        odor_left, odor_right, odor_mean, odor_difference, odor_rate = self._odor_sensors(dt)
        return Sensors(
            target_visible=bool(visual["visible"]),
            target_id=str(visual["id"]),
            target_kind=str(visual["kind"]),
            target_distance=float(visual["distance"]),
            target_bearing=float(visual["azimuth"]),
            target_elevation=float(visual["elevation"]),
            target_bearing_normalized=float(np.clip(float(visual["azimuth"]) / self.half_horizontal_fov, -1.0, 1.0)),
            target_elevation_normalized=float(np.clip(float(visual["elevation"]) / self.half_vertical_fov, -1.0, 1.0)),
            target_angular_size=float(visual["angular_size"]),
            target_angular_rate=float(visual["angular_rate"]),
            target_contrast=float(visual["contrast"]),
            target_occluded=bool(visual["occluded"]),
            target_closing_speed=float(visual["closing_speed"]),
            target_ttc=float(visual["ttc"]),
            obstacle_visible=obstacle_visible,
            obstacle_distance=obstacle_distance,
            obstacle_bearing=obstacle_bearing,
            obstacle_elevation=obstacle_elevation,
            angular_size=angular_size,
            looming_rate=looming_rate,
            obstacle_closing_speed=closing_speed,
            obstacle_ttc=obstacle_ttc,
            odor_left=odor_left,
            odor_right=odor_right,
            odor_mean=odor_mean,
            odor_difference=odor_difference,
            odor_rate=odor_rate,
        )

    def _nearest_food(self) -> tuple[FoodSource | None, float]:
        enabled = [source for source in self.food_sources if source.enabled]
        if not enabled:
            return None, math.inf
        return min(enabled, key=lambda source: math.sqrt((source.x - self.fly.x) ** 2 + (source.y - self.fly.y) ** 2 + (source.z - self.fly.z) ** 2)), min(
            math.sqrt((source.x - self.fly.x) ** 2 + (source.y - self.fly.y) ** 2 + (source.z - self.fly.z) ** 2)
            for source in enabled
        )

    def _sensed_source(self, sensors: Sensors | None) -> FoodSource | None:
        if sensors is None or sensors.target_id in {"none", "manual-target"}:
            return None
        return next(
            (source for source in self.food_sources if source.id == sensors.target_id and source.enabled),
            None,
        )

    def _set_behavior(self, state: str) -> None:
        if state == self.fly.behavior_state:
            return
        self.fly.behavior_state = state
        self._behavior_time = 0.0

    def _update_behavior(self, sensors: Sensors | None, action: bool, dt: float) -> None:
        self._behavior_time += dt
        if action:
            if not self._was_action:
                self._select_escape_height(sensors)
            self._set_behavior("escape")
            return
        if self.fly.behavior_state == "escape" and self._behavior_time < 0.85:
            return
        if self.fly.flight_mode == "perched":
            self._set_behavior("landing")
            return

        candidate = "explore"
        transition_rate = 2.0
        if sensors is not None and sensors.target_visible:
            if sensors.target_distance < 1.25 or sensors.target_angular_size > math.radians(24.0):
                candidate = "landing"
                transition_rate = 12.0
            elif sensors.target_distance < 5.0 or sensors.target_angular_size > math.radians(7.0):
                candidate = "approach"
                transition_rate = 7.0
            else:
                candidate = "visual_lock"
                transition_rate = 5.0
        elif sensors is not None and (sensors.odor_mean > 0.055 or sensors.odor_rate > 0.012):
            candidate = "odor_tracking"
            transition_rate = 3.5

        minimum_dwell = 0.3 if candidate in {"landing", "approach"} else 0.65
        if candidate == self.fly.behavior_state or self._behavior_time < minimum_dwell:
            return
        transition_probability = 1.0 - math.exp(-transition_rate * dt)
        if self.rng.random() < transition_probability:
            self._set_behavior(candidate)

    def _update_microdynamics(self, dt: float) -> np.ndarray:
        root_dt = math.sqrt(dt)
        self._vertical_noise += -0.75 * self._vertical_noise * dt + 0.42 * root_dt * self.rng.normal()
        self._pitch_noise += -1.8 * self._pitch_noise * dt + 0.16 * root_dt * self.rng.normal()
        self._yaw_drift += -0.9 * self._yaw_drift * dt + 0.22 * root_dt * self.rng.normal()
        # Integrate a correlated climb drive, with soft floor/ceiling restoration.
        self._exploration_height += (self._vertical_noise * 0.32 + (2.1 - self._exploration_height) * 0.08) * dt
        self._exploration_height = float(np.clip(self._exploration_height, 0.65, self.glass_height - 0.65))
        local_wind = self._wind_at(self.fly.x, self.fly.y, self.fly.z)
        gust_target = local_wind * 0.18
        self._gust += (gust_target - self._gust) * min(1.0, dt * 0.8)
        self._gust += np.array([
            self.rng.normal(0.0, 0.12),
            self.rng.normal(0.0, 0.08),
            self.rng.normal(0.0, 0.12),
        ]) * root_dt
        return local_wind

    def _world_vertical_direction(self, bearing: float, elevation: float) -> float:
        return (math.sin(elevation) * math.cos(self.fly.pitch)
                + math.cos(elevation) * math.cos(bearing) * math.sin(self.fly.pitch))

    def _select_escape_height(self, sensors: Sensors | None) -> None:
        # DNp01 authorizes the action; this directional choice is NOT neural output.
        self._escape_direction = 0.0
        self._escape_height = self.fly.y
        if sensors is None or not sensors.obstacle_visible:
            return  # Manual neural stimulus without a visible threat: brake only.
        vertical = self._world_vertical_direction(sensors.obstacle_bearing, sensors.obstacle_elevation)
        direction = -1.0 if vertical > 0.08 else 1.0
        floor = self.ground_height_at(self.fly.x, self.fly.z) + self.fly.radius + 0.18
        ceiling = self.glass_height - self.fly.radius - 0.18
        room = ceiling - self.fly.y if direction > 0 else self.fly.y - floor
        if room < 1.55:
            direction *= -1.0
        self._escape_height = float(np.clip(self.fly.y + direction * 1.55, floor, ceiling))
        self._escape_direction = float(np.sign(self._escape_height - self.fly.y))

    def _desired_height(self, sensors: Sensors | None) -> float:
        ground = self.ground_height_at(self.fly.x, self.fly.z)

        state = self.fly.behavior_state
        desired = self._exploration_height
        self._height_policy = "correlated_exploration"
        if state in {"approach", "landing"} and sensors is not None and sensors.target_visible:
            vertical = self._world_vertical_direction(sensors.target_bearing, sensors.target_elevation)
            sensed_height = self.fly.y + vertical * sensors.target_distance
            surface_radius = sensors.target_distance * math.tan(sensors.target_angular_size * 0.5)
            desired = sensed_height + surface_radius + self.fly.radius
            self._height_policy = "visual_surface_approach"
        elif state == "escape":
            desired = self._escape_height
            self._height_policy = "DNp01_gated_direction_policy" if self._escape_direction else "DNp01_brake_hold"
        if self.fly.flight_mode == "taking_off":
            if state != "escape":
                desired = max(desired, ground + 1.0)
                self._height_policy = "takeoff"
        if self.fly.flight_mode == "perched":
            desired = self.fly.y
            self._height_policy = "perched"
        return float(np.clip(desired, ground + self.fly.radius, self.glass_height - self.fly.radius - 0.25))

    def _resolve_collisions(self) -> None:
        position = np.array([self.fly.x, self.fly.y, self.fly.z], dtype=float)
        velocity = np.array([self.fly.vx, self.fly.vy, self.fly.vz], dtype=float)
        current_contacts: set[str] = set()
        new_collision = False
        for collider in self.colliders:
            if not collider.near_point(position, self.fly.radius):
                continue
            center = collider.closest_point(position)
            delta = position - center
            distance = float(np.linalg.norm(delta))
            minimum = self.fly.radius + collider.radius
            if distance >= minimum:
                continue
            current_contacts.add(collider.id)
            if distance > 1e-6:
                normal = delta / distance
            elif collider.shape == "leaf":
                a, b, c = map(np.asarray, collider.vertices)
                normal = np.cross(b - a, c - a)
                normal /= max(float(np.linalg.norm(normal)), 1e-9)
                if float(velocity @ normal) > 0:
                    normal *= -1.0
            else:
                normal = np.array([1.0, 0.0, 0.0])
            position = center + normal * minimum
            inward = float(velocity @ normal)
            if inward < 0.0:
                velocity -= normal * inward
            if collider.id not in self._colliding_ids:
                self.stats["collisions"] += 1
                new_collision = True
        self.fly.x, self.fly.y, self.fly.z = map(float, position)
        self.fly.vx, self.fly.vy, self.fly.vz = map(float, velocity)
        self._colliding_ids = current_contacts
        if new_collision:
            self._collision_recovery = 0.7
            self.fly.flight_mode = "recovering"
        if current_contacts and math.hypot(self.fly.vx, self.fly.vz) > 1e-5:
            self.fly.yaw = math.atan2(self.fly.vz, self.fly.vx)

    def update(
        self,
        dt: float,
        speed: float,
        turn: float,
        max_turn_rate: float,
        action: bool = False,
        sensors: Sensors | None = None,
    ) -> None:
        self.time += dt
        self._mode_time += dt
        self._collision_recovery = max(0.0, self._collision_recovery - dt)
        self._update_behavior(sensors, action, dt)
        local_wind = self._update_microdynamics(dt)

        if self.target.orbit:
            self.target.orbit_phase += dt * 0.35
            self.target.x = 7.0 * math.cos(self.target.orbit_phase)
            self.target.z = 7.0 * math.sin(self.target.orbit_phase)
            self.target.y = 1.0 + 0.35 * math.sin(self.target.orbit_phase * 1.7)

        for source in self.food_sources:
            if source.moving and source.enabled:
                source.phase += dt * 0.22
                source.y += math.sin(source.phase) * dt * 0.08
                source.x += math.cos(source.phase) * dt * 0.18
                source.z += math.sin(source.phase) * dt * 0.18

        if not self.obstacle.enabled and self.time >= self._next_threat_time:
            self.spawn_obstacle(
                float(self.rng.choice([-1.0, 0.0, 1.0])),
                float(self.rng.uniform(7.0, 9.0)),
            )
            self._next_threat_time = self.time + float(self.rng.uniform(8.0, 13.0))

        self._saccade_remaining = max(0.0, self._saccade_remaining - dt)
        self._saccade_cooldown = max(0.0, self._saccade_cooldown - dt)
        if abs(turn) > 0.28 and self._saccade_cooldown <= 0.0:
            self._saccade_sign = math.copysign(1.0, turn)
            self._saccade_remaining = float(self.rng.uniform(0.08, 0.15))
            self._saccade_cooldown = float(self.rng.uniform(0.28, 0.48))
        neural_yaw_rate = turn * max_turn_rate * 0.55
        if self._saccade_remaining > 0.0:
            neural_yaw_rate += self._saccade_sign * max_turn_rate * 1.35
        policy_yaw_rate = self._yaw_drift * (
            0.55 if self.fly.behavior_state in {"explore", "odor_tracking"} else 0.12
        )
        effective_yaw_rate = neural_yaw_rate + policy_yaw_rate
        self.fly.yaw = wrap_angle(self.fly.yaw + effective_yaw_rate * dt)

        speed_scale = {
            "explore": 0.72,
            "odor_tracking": 0.9,
            "visual_lock": 0.82,
            "approach": 1.08,
            "landing": 0.42,
            "escape": 0.0,
        }.get(self.fly.behavior_state, 0.75)
        horizontal_speed = 0.0 if action or self.fly.flight_mode == "perched" else speed * speed_scale
        if self._collision_recovery > 0.0:
            horizontal_speed *= 0.35
        forward_x, forward_z = math.cos(self.fly.yaw), math.sin(self.fly.yaw)
        lateral_x, lateral_z = -forward_z, forward_x
        desired_vx = forward_x * horizontal_speed + self._gust[0] * 0.24
        desired_vz = forward_z * horizontal_speed + self._gust[2] * 0.24
        desired_vx += lateral_x * turn * 0.12
        desired_vz += lateral_z * turn * 0.12
        acceleration = min(1.0, dt * (3.2 if self.fly.behavior_state == "landing" else 4.5))
        self.fly.vx += (desired_vx - self.fly.vx) * acceleration
        self.fly.vz += (desired_vz - self.fly.vz) * acceleration

        desired_height = self._desired_height(sensors)
        ground = self.ground_height_at(self.fly.x, self.fly.z)
        if self.fly.flight_mode == "taking_off" and self.fly.y >= ground + 0.95:
            self.fly.flight_mode = "hovering"
            self._mode_time = 0.0
        if self.fly.behavior_state == "landing" and self.fly.flight_mode not in {"perched", "recovering"}:
            self.fly.flight_mode = "landing"
        elif self.fly.flight_mode == "landing" and self.fly.behavior_state != "landing":
            self.fly.flight_mode = "hovering"
        if self._collision_recovery <= 0.0 and self.fly.flight_mode == "recovering":
            self.fly.flight_mode = "hovering"

        if self.fly.flight_mode == "perched":
            self.fly.vx *= 0.72
            self.fly.vz *= 0.72
            if self._mode_time > 1.2:
                perched = next(
                    (source for source in self.food_sources if source.id == self._perched_source_id),
                    None,
                )
                if perched is not None:
                    perched.enabled = False
                self._perched_source_id = None
                self.fly.flight_mode = "taking_off"
                self._set_behavior("explore")
                self._mode_time = 0.0
                desired_height = max(self._exploration_height, ground + 1.0)

        directional_escape = self.fly.behavior_state == "escape" and self._escape_direction != 0.0
        height_gain, damping = (10.0, 3.2) if directional_escape else (3.2, 2.35)
        vertical_acceleration = (desired_height - self.fly.y) * height_gain - self.fly.vy * damping
        vertical_acceleration += self._vertical_noise * (
            0.8 if self.fly.behavior_state == "explore" else 0.35
        )
        vertical_acceleration += self._gust[1] * 0.45
        clearance = self.fly.y - ground - self.fly.radius
        if 0.0 < clearance < 0.5 and self.fly.flight_mode != "landing":
            vertical_acceleration += (1.0 - clearance / 0.5) * 0.48
        acceleration_limits = (-10.0, 10.0) if directional_escape else (-2.6, 2.9)
        vertical_acceleration = float(np.clip(vertical_acceleration, *acceleration_limits))
        self.fly.vy += vertical_acceleration * dt
        velocity_limits = (-3.0, 3.0) if directional_escape else (-1.2, 1.65)
        self.fly.vy = float(np.clip(self.fly.vy, *velocity_limits))
        self._was_action = action

        self.fly.x += self.fly.vx * dt
        self.fly.y += self.fly.vy * dt
        self.fly.z += self.fly.vz * dt
        margin = self.fly.radius
        boundary_hit = False
        for axis in ("x", "z"):
            value = getattr(self.fly, axis)
            velocity_name = f"v{axis}"
            if abs(value) > self.half_extent - margin:
                setattr(self.fly, axis, math.copysign(self.half_extent - margin, value))
                velocity = getattr(self.fly, velocity_name)
                if velocity * value > 0.0:
                    setattr(self.fly, velocity_name, -velocity * 0.65)
                    boundary_hit = True
        if boundary_hit:
            self.fly.yaw = math.atan2(self.fly.vz, self.fly.vx)
            self.stats["collisions"] += 1
            self._collision_recovery = 0.7
            self.fly.flight_mode = "recovering"
        ground = self.ground_height_at(self.fly.x, self.fly.z)
        floor = ground + margin
        if self.fly.y < floor:
            self.fly.y = floor
            self.fly.vy = max(0.0, self.fly.vy)
        elif self.fly.y > self.glass_height - margin:
            self.fly.y = self.glass_height - margin
            self.fly.vy = min(0.0, self.fly.vy)
            self.stats["collisions"] += 1
            self._collision_recovery = 0.7
            self.fly.flight_mode = "recovering"

        self._resolve_collisions()
        horizontal = math.hypot(self.fly.vx, self.fly.vz)
        target_pitch = math.atan2(self.fly.vy, max(horizontal, 0.2)) + self._pitch_noise
        self.fly.pitch += (
            float(np.clip(target_pitch, -0.58, 0.58)) - self.fly.pitch
        ) * min(1.0, dt * 4.2)
        wing_rate = 19.0 + horizontal * 8.0 + abs(self.fly.vy) * 5.0
        if action:
            wing_rate += 18.0
        self.fly.flight_phase += dt * wing_rate

        sensed_source = self._sensed_source(sensors)
        if sensed_source is not None:
            source_distance = math.sqrt(
                (sensed_source.x - self.fly.x) ** 2
                + (sensed_source.y - self.fly.y) ** 2
                + (sensed_source.z - self.fly.z) ** 2
            )
            if source_distance < sensed_source.radius + self.fly.radius + 0.18:
                if sensed_source.id not in self._found_ids:
                    self._found_ids.add(sensed_source.id)
                    self.stats["sources_reached"] += 1
                    self.stats["fruits_found"] += int(sensed_source.kind == "fruit")
                if self.fly.flight_mode == "landing" and abs(self.fly.vy) < 0.35:
                    self.fly.flight_mode = "perched"
                    self._perched_source_id = sensed_source.id
                    self._mode_time = 0.0
                    self.fly.vy = 0.0
                    self.stats["landings"] += 1

        if self.obstacle.enabled:
            self.obstacle.age += dt
            self.obstacle.x += self.obstacle.vx * dt
            self.obstacle.y += self.obstacle.vy * dt
            self.obstacle.z += self.obstacle.vz * dt
            distance = math.sqrt(
                (self.obstacle.x - self.fly.x) ** 2
                + (self.obstacle.y - self.fly.y) ** 2
                + (self.obstacle.z - self.fly.z) ** 2
            )
            if distance < self.obstacle.radius + self.fly.radius:
                self._threat_contact = True
                self.stats["threat_contacts"] += 1
                self.obstacle.enabled = False
            elif self.obstacle.age > 3.2 or abs(self.obstacle.x) > 17.0 or abs(self.obstacle.z) > 17.0:
                if not self._threat_contact:
                    self.stats["threats_avoided"] += 1
                self.obstacle.enabled = False

        self.motion_policy = {
            "behavior_state": self.fly.behavior_state,
            "desired_height": desired_height,
            "height_policy": self._height_policy,
            "escape_trigger": "DNp01" if self.fly.behavior_state == "escape" else "none",
            "escape_direction_policy": self._escape_direction if self.fly.behavior_state == "escape" else 0.0,
            "effective_speed": horizontal_speed,
            "neural_yaw_rate": neural_yaw_rate,
            "policy_yaw_rate": policy_yaw_rate,
            "effective_yaw_rate": effective_yaw_rate,
            "vertical_noise": self._vertical_noise,
            "saccade_active": self._saccade_remaining > 0.0,
            "gust_x": float(self._gust[0]),
            "gust_y": float(self._gust[1]),
            "gust_z": float(self._gust[2]),
            "wind_x": float(local_wind[0]),
            "wind_y": float(local_wind[1]),
            "wind_z": float(local_wind[2]),
        }
