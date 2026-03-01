from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from Box2D import (
    b2CircleShape,
    b2ContactListener,
    b2FixtureDef,
    b2PolygonShape,
    b2RevoluteJointDef,
    b2Vec2,
    b2World,
)
from gymnasium import spaces

from .curriculum import CurriculumScheduler
from .reward import ContactFeature, compute_reward


@dataclass
class EnvConfig:
    max_steps: int = 300
    control_dt: float = 0.02
    substeps: int = 4
    shape_curriculum: bool = True
    success_height: float = 0.22
    success_hold_steps: int = 25
    drop_height: float = 0.03
    max_joint_delta: float = 0.10
    motor_kp: float = 12.0
    max_motor_speed: float = 6.0
    max_motor_torque: float = 30.0
    palm_height: float = 0.32
    finger_link_lengths: tuple[float, float] = (0.13, 0.11)
    finger_base_x: tuple[float, float] = (-0.14, 0.14)
    reachability_reset: bool = True


def _vec2(arr: np.ndarray | tuple[float, float]) -> b2Vec2:
    return b2Vec2(float(arr[0]), float(arr[1]))


class _ContactTracker(b2ContactListener):
    def __init__(self, num_fingers: int) -> None:
        super().__init__()
        self.num_fingers = num_fingers
        self._default()

    def _default(self) -> None:
        self.data = {i: self._new_item() for i in range(self.num_fingers)}

    @staticmethod
    def _new_item() -> dict[str, Any]:
        return {
            "contact": False,
            "normal_impulse": 0.0,
            "tangent_impulse": 0.0,
            "normal": np.array([0.0, 1.0], dtype=np.float32),
            "point": np.array([0.0, 0.0], dtype=np.float32),
            "count": 0,
            "force_dir": np.array([0.0, 1.0], dtype=np.float32),
        }

    def reset_step(self) -> None:
        self._default()

    @staticmethod
    def _tags(contact: Any) -> tuple[Any, Any]:
        return contact.fixtureA.userData, contact.fixtureB.userData

    def _extract(self, contact: Any) -> tuple[int | None, bool, np.ndarray, np.ndarray]:
        tag_a, tag_b = self._tags(contact)
        if not isinstance(tag_a, tuple) or not isinstance(tag_b, tuple):
            return None, False, np.array([0.0, 1.0]), np.array([0.0, 0.0])
        if tag_a[0] == "finger" and tag_b[0] == "object":
            finger_idx = int(tag_a[1])
            object_is_b = True
        elif tag_b[0] == "finger" and tag_a[0] == "object":
            finger_idx = int(tag_b[1])
            object_is_b = False
        else:
            return None, False, np.array([0.0, 1.0]), np.array([0.0, 0.0])
        manifold = contact.manifold
        if manifold is None or manifold.pointCount <= 0:
            return None, False, np.array([0.0, 1.0]), np.array([0.0, 0.0])
        wm = contact.worldManifold
        normal = np.array([wm.normal[0], wm.normal[1]], dtype=np.float32)
        point = np.array([wm.points[0][0], wm.points[0][1]], dtype=np.float32)
        return finger_idx, object_is_b, normal, point

    def BeginContact(self, contact: Any) -> None:  # noqa: N802
        finger_idx, object_is_b, normal, point = self._extract(contact)
        if finger_idx is None:
            return
        item = self.data[finger_idx]
        item["contact"] = True
        item["normal"] = normal
        item["point"] = point
        item["count"] += 1
        item["force_dir"] = normal if object_is_b else -normal

    def PostSolve(self, contact: Any, impulse: Any) -> None:  # noqa: N802
        finger_idx, object_is_b, normal, point = self._extract(contact)
        if finger_idx is None:
            return
        item = self.data[finger_idx]
        item["contact"] = True
        item["normal"] = normal
        item["point"] = point
        item["count"] += 1
        item["force_dir"] = normal if object_is_b else -normal
        item["normal_impulse"] += float(sum(impulse.normalImpulses))
        item["tangent_impulse"] += float(sum(abs(v) for v in impulse.tangentImpulses))


class DexGrasp2DEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 300,
        control_dt: float = 0.02,
        substeps: int = 4,
        num_fingers: int = 2,
        shape_curriculum: bool = True,
        reward_weights: dict[str, float] | None = None,
        success_height: float = 0.22,
        success_hold_steps: int = 25,
        drop_height: float = 0.03,
        palm_height: float = 0.32,
        finger_link_lengths: tuple[float, float] = (0.13, 0.11),
        finger_base_x: tuple[float, float] = (-0.14, 0.14),
        reachability_reset: bool = True,
        screen_width: int = 800,
        screen_height: int = 600,
        pixels_per_meter: float = 300.0,
        show_hud: bool = True,
    ) -> None:
        if num_fingers != 2:
            raise ValueError("DexGrasp2DEnv v0 supports exactly 2 fingers.")
        if len(finger_base_x) != num_fingers:
            raise ValueError("finger_base_x length must match num_fingers.")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.config = EnvConfig(
            max_steps=max_steps,
            control_dt=control_dt,
            substeps=substeps,
            shape_curriculum=shape_curriculum,
            success_height=success_height,
            success_hold_steps=success_hold_steps,
            drop_height=drop_height,
            palm_height=palm_height,
            finger_link_lengths=finger_link_lengths,
            finger_base_x=finger_base_x,
            reachability_reset=reachability_reset,
        )
        self.reward_weights = reward_weights or {
            "w_contact": 0.25,
            "w_inward": 0.20,
            "w_wrench": 0.25,
            "w_still": 0.15,
            "w_lift": 0.15,
            "w_reg": 0.05,
        }
        self.num_fingers = num_fingers
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * self.num_fingers,), dtype=np.float32
        )
        obs_size = (2 * self.num_fingers) + (2 * self.num_fingers) + (2 * self.num_fingers) + (2 * self.num_fingers) + 6 + (4 * self.num_fingers) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self._np_random = np.random.default_rng(0)
        self._tracker = _ContactTracker(num_fingers=self.num_fingers)
        self._world: b2World | None = None
        self._ground = None
        self._palm = None
        self._fingers: list[Any] = []
        self._distal_links: list[Any] = []
        self._joints: list[Any] = []
        self._object = None
        self._target_angles = np.zeros(2 * self.num_fingers, dtype=np.float32)
        self._prev_action = np.zeros(2 * self.num_fingers, dtype=np.float32)
        self._episode_steps = 0
        self._hold_counter = 0
        self._start_object_height = 0.0
        self._last_success = False
        self._scheduler = CurriculumScheduler(enabled=shape_curriculum)
        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0

        self._screen_width = screen_width
        self._screen_height = screen_height
        self._ppm = pixels_per_meter
        self._show_hud = show_hud
        self._pygame: Any = None
        self._clock: Any = None
        self._screen: Any = None
        self._canvas: Any = None
        self._font: Any = None
        self._colors = {
            "bg": (245, 247, 250),
            "ground": (70, 80, 95),
            "palm": (40, 50, 65),
            "finger": (41, 128, 185),
            "object": (230, 126, 34),
            "contact": (220, 53, 69),
            "text": (33, 37, 41),
        }

    def _init_renderer(self) -> None:
        if self._pygame is not None and self._canvas is not None:
            return
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "PyGame is required for rendering. Install with `uv add pygame`."
            ) from exc
        self._pygame = pygame
        if self.render_mode == "human":
            pygame.display.init()
            try:
                self._screen = pygame.display.set_mode(
                    (self._screen_width, self._screen_height)
                )
            except pygame.error as exc:
                raise RuntimeError(
                    "Failed to open a display for human rendering. "
                    "Use render_mode='rgb_array' in headless environments."
                ) from exc
            pygame.display.set_caption("DexGrasp2D-v0")
            self._clock = pygame.time.Clock()
        self._canvas = pygame.Surface((self._screen_width, self._screen_height))
        pygame.font.init()
        self._font = pygame.font.Font(None, 22)

    def _world_to_screen(self, point: tuple[float, float] | list[float] | np.ndarray) -> tuple[int, int]:
        x = int(self._screen_width * 0.5 + float(point[0]) * self._ppm)
        y = int(self._screen_height - float(point[1]) * self._ppm)
        return (x, y)

    def _draw_body_fixture(self, surface: Any, body: Any, color: tuple[int, int, int]) -> None:
        assert self._pygame is not None
        for fixture in body.fixtures:
            shape = fixture.shape
            if isinstance(shape, b2CircleShape):
                center_world = body.GetWorldPoint(shape.pos)
                center = self._world_to_screen(center_world)
                radius = max(1, int(shape.radius * self._ppm))
                self._pygame.draw.circle(surface, color, center, radius)
            elif isinstance(shape, b2PolygonShape):
                verts_world = [body.GetWorldPoint(v) for v in shape.vertices]
                verts = [self._world_to_screen(v) for v in verts_world]
                if len(verts) >= 3:
                    self._pygame.draw.polygon(surface, color, verts)

    def _draw_hud(self, surface: Any) -> None:
        if not self._show_hud or self._font is None:
            return
        reward_terms = self._last_info.get("reward_terms", {})
        lines = [
            f"step: {self._episode_steps}",
            f"stage: {self._last_info.get('curriculum_stage', 0)}",
            f"contacts: {self._last_info.get('contact_count', 0)}",
            f"reward: {self._last_reward:.3f}",
            f"success: {self._last_info.get('is_success', False)}",
            f"wrench: {reward_terms.get('wrench', 0.0):.3f}",
        ]
        y = 10
        for line in lines:
            text = self._font.render(line, True, self._colors["text"])
            surface.blit(text, (10, y))
            y += 22

    def _draw_scene(self, surface: Any) -> None:
        assert self._pygame is not None
        surface.fill(self._colors["bg"])
        if self._ground is not None:
            self._draw_body_fixture(surface, self._ground, self._colors["ground"])
        for finger in self._fingers:
            self._draw_body_fixture(surface, finger, self._colors["finger"])
        if self._object is not None:
            self._draw_body_fixture(surface, self._object, self._colors["object"])
        if self._palm is not None:
            palm_center = self._world_to_screen((self._palm.position[0], self._palm.position[1]))
            self._pygame.draw.circle(surface, self._colors["palm"], palm_center, 8)
        for i in range(self.num_fingers):
            item = self._tracker.data[i]
            if not item["contact"]:
                continue
            point = self._world_to_screen(item["point"])
            self._pygame.draw.circle(surface, self._colors["contact"], point, 4)
        self._draw_hud(surface)

    def _build_world(self) -> None:
        self._world = b2World(gravity=(0.0, -9.8), doSleep=True)
        self._world.contactListener = self._tracker
        self._ground = self._world.CreateStaticBody(
            position=(0.0, 0.0),
            shapes=b2PolygonShape(box=(1.0, 0.02, (0.0, -0.02), 0.0)),
        )
        self._palm = self._world.CreateStaticBody(position=(0.0, self.config.palm_height))
        anchors = [(x, 0.0) for x in self.config.finger_base_x]
        base_angles = [0.25, -0.25]
        l1, l2 = self.config.finger_link_lengths
        self._fingers = []
        self._distal_links = []
        self._joints = []
        for i, (anchor, angle) in enumerate(zip(anchors, base_angles, strict=True)):
            prox = self._world.CreateDynamicBody(
                position=(anchor[0], self.config.palm_height - l1 * 0.5),
                angle=angle,
                linearDamping=1.4,
                angularDamping=1.6,
            )
            prox_fixture = b2FixtureDef(
                shape=b2PolygonShape(box=(0.020, l1 * 0.5)),
                density=1.0,
                friction=1.2,
                restitution=0.0,
            )
            prox.CreateFixture(prox_fixture)
            prox.fixtures[0].userData = ("finger", i)

            distal = self._world.CreateDynamicBody(
                position=(anchor[0], self.config.palm_height - l1 - l2 * 0.5),
                angle=angle,
                linearDamping=1.4,
                angularDamping=1.6,
            )
            dist_fixture = b2FixtureDef(
                shape=b2PolygonShape(box=(0.018, l2 * 0.5)),
                density=1.0,
                friction=1.3,
                restitution=0.0,
            )
            distal.CreateFixture(dist_fixture)
            distal.fixtures[0].userData = ("finger", i)

            prox_joint = b2RevoluteJointDef(
                bodyA=self._palm,
                bodyB=prox,
                localAnchorA=anchor,
                localAnchorB=(0.0, l1 * 0.5),
                enableLimit=True,
                lowerAngle=-1.2,
                upperAngle=1.0,
                enableMotor=True,
                maxMotorTorque=self.config.max_motor_torque,
                motorSpeed=0.0,
            )
            dist_joint = b2RevoluteJointDef(
                bodyA=prox,
                bodyB=distal,
                localAnchorA=(0.0, -l1 * 0.5),
                localAnchorB=(0.0, l2 * 0.5),
                enableLimit=True,
                lowerAngle=-1.4,
                upperAngle=1.2,
                enableMotor=True,
                maxMotorTorque=self.config.max_motor_torque,
                motorSpeed=0.0,
            )
            self._fingers.extend([prox, distal])
            self._distal_links.append(distal)
            self._joints.append(self._world.CreateJoint(prox_joint))
            self._joints.append(self._world.CreateJoint(dist_joint))

    def _sample_reachable_object_position(self) -> tuple[float, float]:
        l1, l2 = self.config.finger_link_lengths
        r_max = (l1 + l2) * 0.95
        y_min, y_max = 0.08, 0.13
        x_min = min(self.config.finger_base_x) - 0.02
        x_max = max(self.config.finger_base_x) + 0.02
        for _ in range(64):
            x = float(self._np_random.uniform(x_min, x_max))
            y = float(self._np_random.uniform(y_min, y_max))
            dists = [
                np.hypot(x - ax, y - self.config.palm_height)
                for ax in self.config.finger_base_x
            ]
            if min(dists) <= r_max:
                return x, y
        return 0.0, 0.10

    def _spawn_object(self) -> None:
        assert self._world is not None
        if self._object is not None:
            self._world.DestroyBody(self._object)
            self._object = None
        shape = self._scheduler.sample_shape(self._np_random)
        if self.config.reachability_reset:
            x, y = self._sample_reachable_object_position()
        else:
            x = float(self._np_random.uniform(-0.10, 0.10))
            y = float(self._np_random.uniform(0.08, 0.11))
        density = float(self._np_random.uniform(0.6, 1.4))
        friction = float(self._np_random.uniform(0.6, 1.3))
        body = self._world.CreateDynamicBody(position=(x, y))
        if shape == "circle":
            radius = float(self._np_random.uniform(0.045, 0.065))
            body.CreateFixture(
                shape=b2CircleShape(radius=radius),
                density=density,
                friction=friction,
                restitution=0.0,
            )
        elif shape == "box":
            hx = float(self._np_random.uniform(0.040, 0.070))
            hy = float(self._np_random.uniform(0.040, 0.070))
            body.CreateFixture(
                shape=b2PolygonShape(box=(hx, hy)),
                density=density,
                friction=friction,
                restitution=0.0,
            )
        else:
            n = int(self._np_random.integers(3, 7))
            angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            radii = self._np_random.uniform(0.04, 0.07, size=n)
            vertices = [(float(r * np.cos(a)), float(r * np.sin(a))) for r, a in zip(radii, angles)]
            body.CreateFixture(
                shape=b2PolygonShape(vertices=vertices),
                density=density,
                friction=friction,
                restitution=0.0,
            )
        body.fixtures[0].userData = ("object", 0)
        body.angle = float(self._np_random.uniform(-0.2, 0.2))
        self._object = body
        self._start_object_height = float(body.position[1])

    def _finger_observation(self) -> np.ndarray:
        joint_angles = np.array([j.angle for j in self._joints], dtype=np.float32)
        joint_speeds = np.array([j.speed for j in self._joints], dtype=np.float32)
        tips: list[float] = []
        tip_vels: list[float] = []
        l2 = self.config.finger_link_lengths[1]
        for finger in self._distal_links:
            tip = finger.GetWorldPoint(localPoint=(0.0, -l2 * 0.5))
            vel = finger.GetLinearVelocityFromWorldPoint(worldPoint=tip)
            tips.extend([float(tip[0]), float(tip[1])])
            tip_vels.extend([float(vel[0]), float(vel[1])])
        return np.concatenate(
            [
                joint_angles,
                joint_speeds,
                np.array(tips, dtype=np.float32),
                np.array(tip_vels, dtype=np.float32),
            ]
        )

    def _contact_features(self) -> tuple[np.ndarray, list[ContactFeature]]:
        assert self._object is not None
        object_pos = np.array(self._object.position, dtype=np.float32)
        items: list[float] = []
        features: list[ContactFeature] = []
        for i in range(self.num_fingers):
            item = self._tracker.data[i]
            contact = bool(item["contact"])
            point = np.array(item["point"], dtype=np.float32)
            force_dir = np.array(item["force_dir"], dtype=np.float32)
            normal_force = float(item["normal_impulse"]) / max(self.config.control_dt, 1e-4)
            tangent_force = float(item["tangent_impulse"]) / max(self.config.control_dt, 1e-4)
            slip = float(tangent_force / (normal_force + 1e-3))
            inward = 0.0
            if contact:
                inward_vec = object_pos - point
                inward_norm = float(np.linalg.norm(inward_vec) + 1e-6)
                inward = float(np.dot(force_dir, inward_vec / inward_norm))
            features.append(
                ContactFeature(
                    contact=contact,
                    point=point,
                    force_dir=force_dir,
                    inward_alignment=inward,
                    normal_force=normal_force,
                    slip=slip,
                )
            )
            items.extend(
                [
                    1.0 if contact else 0.0,
                    normal_force * 0.01,
                    inward,
                    slip,
                ]
            )
        return np.array(items, dtype=np.float32), features

    def _build_obs(self) -> tuple[np.ndarray, list[ContactFeature]]:
        assert self._object is not None
        finger_obs = self._finger_observation()
        obj = np.array(
            [
                self._object.position[0],
                self._object.position[1],
                self._object.angle,
                self._object.linearVelocity[0],
                self._object.linearVelocity[1],
                self._object.angularVelocity,
            ],
            dtype=np.float32,
        )
        contact_obs, features = self._contact_features()
        stage = np.array([float(self._scheduler.current_stage())], dtype=np.float32)
        obs = np.concatenate([finger_obs, obj, contact_obs, stage]).astype(np.float32)
        return obs, features

    def _set_motor_targets(self) -> None:
        for i, joint in enumerate(self._joints):
            err = float(self._target_angles[i] - joint.angle)
            speed = np.clip(
                self.config.motor_kp * err,
                -self.config.max_motor_speed,
                self.config.max_motor_speed,
            )
            joint.motorSpeed = float(speed)
            joint.maxMotorTorque = self.config.max_motor_torque

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._tracker.reset_step()
        self._episode_steps = 0
        self._hold_counter = 0
        self._prev_action = np.zeros(2 * self.num_fingers, dtype=np.float32)
        default_targets: list[float] = []
        for base in [0.25, -0.25]:
            default_targets.extend([base, 0.6])
        self._target_angles = np.array(default_targets, dtype=np.float32)
        if self._world is not None:
            self._world.contactListener = None
            self._world = None
            self._ground = None
            self._palm = None
            self._fingers = []
            self._distal_links = []
            self._joints = []
            self._object = None
        self._build_world()
        self._spawn_object()
        obs, _ = self._build_obs()
        info = {
            "reward_terms": {},
            "is_success": False,
            "contact_count": 0,
            "stability_score": 0.0,
            "curriculum_stage": self._scheduler.current_stage(),
        }
        self._last_info = info
        self._last_reward = 0.0
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._world is not None
        assert self._object is not None
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )
        self._target_angles = np.clip(
            self._target_angles + self.config.max_joint_delta * action,
            -1.2,
            1.2,
        )
        self._set_motor_targets()
        dt = self.config.control_dt / self.config.substeps
        self._tracker.reset_step()
        for _ in range(self.config.substeps):
            self._world.Step(dt, 8, 3)
        self._episode_steps += 1

        obs, contact_features = self._build_obs()
        object_pos = np.array(self._object.position, dtype=np.float32)
        lift_progress = (object_pos[1] - self._start_object_height) / max(
            self.config.success_height - self._start_object_height, 1e-6
        )
        dropped = bool(object_pos[1] < self.config.drop_height)
        lifted = bool(object_pos[1] > self.config.success_height)
        obj_lin_vel = np.array(self._object.linearVelocity, dtype=np.float32)
        obj_ang = float(self._object.angularVelocity)
        stable = bool(np.linalg.norm(obj_lin_vel) < 0.20 and abs(obj_ang) < 1.5)
        if lifted and stable:
            self._hold_counter += 1
        else:
            self._hold_counter = 0
        success = self._hold_counter >= self.config.success_hold_steps

        breakdown = compute_reward(
            contacts=contact_features,
            object_pos=object_pos,
            object_lin_vel=obj_lin_vel,
            object_ang_vel=obj_ang,
            lift_progress=lift_progress,
            action=action,
            prev_action=self._prev_action,
            reward_weights=self.reward_weights,
            success=success,
            dropped=dropped,
            expected_contacts=self.num_fingers,
        )
        self._prev_action = action.copy()
        terminated = dropped or success
        truncated = self._episode_steps >= self.config.max_steps
        if terminated or truncated:
            self._scheduler.update(success=success)
            self._last_success = success
        info = {
            "reward_terms": {
                "contact": breakdown.contact,
                "inward": breakdown.inward,
                "wrench": breakdown.wrench,
                "still": breakdown.still,
                "lift": breakdown.lift,
                "reg": breakdown.reg,
                "terminal": breakdown.terminal,
            },
            "is_success": success,
            "contact_count": int(sum(c.contact for c in contact_features)),
            "stability_score": breakdown.stability_score,
            "curriculum_stage": self._scheduler.current_stage(),
        }
        self._last_info = info
        self._last_reward = breakdown.total
        if self.render_mode == "human":
            self.render()
        return obs, breakdown.total, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            raise RuntimeError(
                "render() called with render_mode=None. Create env with "
                "render_mode='human' or 'rgb_array'."
            )
        self._init_renderer()
        assert self._pygame is not None
        assert self._canvas is not None
        self._draw_scene(self._canvas)
        if self.render_mode == "human":
            assert self._screen is not None
            self._screen.blit(self._canvas, (0, 0))
            self._pygame.display.flip()
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self.close()
                    break
            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])
            return None
        frame = self._pygame.surfarray.array3d(self._canvas)
        return np.transpose(frame, (1, 0, 2)).astype(np.uint8, copy=False)

    def close(self) -> None:
        if self._pygame is not None:
            if self._screen is not None:
                self._pygame.display.quit()
            self._pygame.font.quit()
        self._screen = None
        self._canvas = None
        self._clock = None
        self._font = None
        self._pygame = None
        self._world = None
