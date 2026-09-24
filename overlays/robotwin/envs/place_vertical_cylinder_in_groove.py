"""Phone-view-aligned single-Panthera vertical cylinder insertion task.

One six-axis Panthera top-down-grasps an upright yellow cylinder, moves it over a
shallow yellow block/socket, inserts it vertically, releases it and retracts.
The public observation/action contract remains six arm joints plus one
normalized gripper value.  No scripted attachment is used.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d

from .place_cylinder_in_groove import (
    ROBOT_SOURCE_COMMIT,
    ROBOTWIN_SOURCE_COMMIT,
    TABLE_HEIGHT_M,
    place_cylinder_in_groove,
)
from .panthera_release_reward import ReleaseEffectState
from .utils import Action, Actor, ArmTag, create_box


TASK_SCHEMA_VERSION = 4


def _profiles_root() -> Path:
    """Where the asset profiles live, seen from inside the runtime.

    The task runs from an assembled runtime, whose ``assets/profiles`` comes
    from the overlay, so the profiles sit beside this module's own package.
    ``PANTHERA_ASSET_PROFILES`` overrides it for a checkout that is laid out
    some other way.
    """
    override = os.environ.get("PANTHERA_ASSET_PROFILES")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "assets" / "profiles"


def _object_profile(name: str):
    """Read one object's profile through the access layer, not by hand.

    These numbers used to be literals here and in ``place_cylinder_in_groove``
    at the same time, with the profiles written during the restructure and
    never read: ``assets.load_object`` had no callers at all.  The colour was
    the worst of it -- nothing recorded it, so a change to it was invisible to
    every consumer, and a dataset ended up showing one colour while its own
    instruction named another.
    """
    packages = os.environ.get("PANTHERA_PACKAGES")
    if not packages:
        workspace = os.environ.get("PANTHERA_VLA_ROOT", "/data/lyy/panthera-vla")
        packages = str(Path(workspace) / "packages")
    sim = str(Path(packages) / "panthera_sim")
    if sim not in sys.path:
        sys.path.insert(0, sim)
    from assets import load_object

    return load_object(_profiles_root(), name)


CYLINDER_PROFILE = _object_profile("panthera_cylinder")
SOCKET_PROFILE = _object_profile("panthera_socket")

CYLINDER_RADIUS_M = float(CYLINDER_PROFILE.geometry["radius_m"])
CYLINDER_HALF_HEIGHT_M = float(CYLINDER_PROFILE.geometry["half_height_m"])
CYLINDER_MASS_KG = float(CYLINDER_PROFILE.physics["mass_kg"])
CYLINDER_BASE_COLOR = list(CYLINDER_PROFILE.appearance["base_color"])
SOCKET_BASE_COLOR = list(SOCKET_PROFILE.appearance["base_color"])
SOCKET_INNER_HALF_WIDTH_M = float(SOCKET_PROFILE.geometry["inner_half_width_m"])
SOCKET_WALL_HALF_WIDTH_M = 0.009
SOCKET_WALL_HALF_HEIGHT_M = 0.020
SOCKET_FLOOR_HALF_HEIGHT_M = 0.030
SETTLE_SIMULATION_STEPS = 250
PREGRASP_OPENING = 0.90
CYLINDER_GRASP_OPENING = 0.40
NOMINAL_GRIPPER_TO_CYLINDER_CENTER_M = 0.120
# With the reviewed top-down grasp, the cylinder center settles about 4--5 mm
# toward world +Y from the measured end-effector origin. This fixed tool/object
# calibration is applied to the commanded EE target; it is not online simulator
# object-pose feedback.
NOMINAL_CYLINDER_CENTER_FROM_EE_Y_M = 0.0045
TERMINAL_ASSIST_ALIGN_SEGMENT_M = 0.005
TERMINAL_ASSIST_INSERT_SEGMENT_M = 0.0025
TERMINAL_ASSIST_HOVER_OFFSET_M = 0.050
TERMINAL_ASSIST_DEPARTURE_LIFT_M = 0.030
TERMINAL_ASSIST_APPROACH_Y_OFFSET_M = 0.085
TERMINAL_ASSIST_HANDOFF_SETTLE_STEPS = 250
TERMINAL_ASSIST_ALIGN_SETTLE_STEPS = 25
TERMINAL_ASSIST_INSERT_SETTLE_STEPS = 50
TERMINAL_ASSIST_PRE_RELEASE_SETTLE_STEPS = 250
TERMINAL_ASSIST_POST_RELEASE_SETTLE_STEPS = 250
GRASP_PROXY_MIN_OPENING = 0.60
GRASP_PROXY_MAX_OPENING = 0.75
GRASP_PROXY_REQUIRED_CONTROL_STEPS = 50
TERMINAL_ASSIST_RELEASE_Z_MIN_M = 0.008

# Local tool X is the approach direction and local tool Y is the finger-closing
# direction in RoboTwin's Panthera wrapper.  This rotation maps them to world
# -Z and -X respectively: the wrist descends from above while the fingers close
# horizontally around the upright cylinder.  This is the Panthera orientation
# already exercised by the horizontal-cylinder oracle and avoids an unreachable
# wrist pose at table height.
TOP_DOWN_GRASP_QUATERNION_WXYZ = [
    0.5,
    -0.5,
    0.5,
    0.5,
]

# A local-X cylinder becomes vertical when local X is rotated onto world +Z.
UPRIGHT_CYLINDER_QUATERNION_WXYZ = [
    math.sqrt(0.5),
    0.0,
    -math.sqrt(0.5),
    0.0,
]


def _frame(rotation: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result.tolist()


def _vertical_cylinder_actor_data() -> dict:
    identity = np.eye(4, dtype=float).tolist()
    return {
        "center": [0.0, 0.0, 0.0],
        "extents": [
            2.0 * CYLINDER_HALF_HEIGHT_M,
            2.0 * CYLINDER_RADIUS_M,
            2.0 * CYLINDER_RADIUS_M,
        ],
        "scale": [1.0, 1.0, 1.0],
        "target_pose": [identity],
        "contact_points_pose": [_frame(np.eye(3), np.zeros(3))],
        "functional_matrix": [identity],
        "orientation_point": [identity],
        "contact_points_group": [[0]],
        "contact_points_mask": [True],
        "contact_points_description": ["upright cylinder center"],
        "target_point_description": ["upright cylinder center"],
    }


def _create_upright_cylinder(task, pose: sapien.Pose) -> sapien.Entity:
    entity = sapien.Entity()
    # Retain the common semantic name so inherited telemetry and replay tools
    # count contacts without task-specific branches.
    entity.set_name("panthera_cylinder")
    entity.set_pose(pose)

    material = task.scene.create_physical_material(
        float(task.physics_parameters["static_friction"]),
        float(task.physics_parameters["dynamic_friction"]),
        0.0,
    )
    rigid = sapien.physx.PhysxRigidDynamicComponent()
    rigid.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=CYLINDER_RADIUS_M,
            half_length=CYLINDER_HALF_HEIGHT_M,
            material=material,
        )
    )
    render = sapien.render.RenderBodyComponent()
    render.attach(
        sapien.render.RenderShapeCylinder(
            radius=CYLINDER_RADIUS_M,
            half_length=CYLINDER_HALF_HEIGHT_M,
            material=sapien.render.RenderMaterial(
                base_color=CYLINDER_BASE_COLOR
            ),
        )
    )
    entity.add_component(rigid)
    entity.add_component(render)
    entity.set_pose(pose)
    task.scene.add_entity(entity)
    return entity


class place_vertical_cylinder_in_groove(place_cylinder_in_groove):
    """Single-arm upright peg insertion aligned to the archived phone view."""

    def setup_demo(self, **kwargs):
        """Force the reviewed top-down grasp unless explicitly configured."""
        gripper_static_friction = float(kwargs.get("gripper_static_friction", 8.0))
        gripper_dynamic_friction = float(kwargs.get("gripper_dynamic_friction", 6.0))
        initial_gripper_opening = kwargs.get("initial_gripper_opening")
        initial_gripper_settle_steps = int(
            kwargs.get("initial_gripper_settle_steps", 250)
        )
        configured = kwargs.get("grasp_quaternion_wxyz")
        if configured is None:
            kwargs["grasp_quaternion_wxyz"] = TOP_DOWN_GRASP_QUATERNION_WXYZ
        super().setup_demo(**kwargs)
        self.reset_policy_reward_state()
        self.physics_parameters["gripper_static_friction"] = gripper_static_friction
        self.physics_parameters["gripper_dynamic_friction"] = gripper_dynamic_friction
        pad_material = self.scene.create_physical_material(
            gripper_static_friction,
            gripper_dynamic_friction,
            0.0,
        )
        finger_links = 0
        for link in self.robot.left_entity.get_links():
            if link.get_name() not in {"L_finger", "R_finger"}:
                continue
            finger_links += 1
            for shape in link.get_collision_shapes():
                shape.set_physical_material(pad_material)
        if finger_links != 2:
            raise ValueError(f"expected two Panthera finger links, got {finger_links}")
        if initial_gripper_opening is not None:
            initial_gripper_opening = float(initial_gripper_opening)
            if not math.isfinite(initial_gripper_opening):
                raise ValueError("initial_gripper_opening must be finite")
            if not 0.0 <= initial_gripper_opening <= 1.0:
                raise ValueError("initial_gripper_opening must be in [0, 1]")
            if initial_gripper_settle_steps < 0:
                raise ValueError("initial_gripper_settle_steps must be nonnegative")
            # The demonstration enters arm motion with the normalized gripper
            # already at PREGRASP_OPENING.  This optional evaluation preamble
            # reproduces that state before the policy receives its first frame.
            # It advances simulation time directly and records no policy step.
            self.robot.set_gripper(
                initial_gripper_opening,
                "left",
                gripper_eps=0.0,
            )
            self._advance_physics(initial_gripper_settle_steps)
            self.initial_gripper_opening = initial_gripper_opening
            self.initial_gripper_settle_steps = initial_gripper_settle_steps

    def reset_policy_reward_state(self) -> None:
        """Reset release-settling latches without changing simulator state."""
        self._rl_release_elapsed_physics_steps = 0
        self._rl_stable_success_checks = 0
        self._rl_success_latched = False
        self._rl_last_effect_physics_step = int(self.simulation_step_count)
        self._reset_policy_episode_metrics()

    def policy_release_effect_state(
        self, *, update_settle: bool
    ) -> ReleaseEffectState:
        """Measure privileged task effects for reward, never policy input.

        Success is deliberately stricter than one instantaneous metrics check:
        the cylinder must remain in a valid released state for at least one
        simulated second, followed by three consecutive stable 50 Hz checks.
        """
        cylinder_pose = self.cylinder.get_pose()
        cylinder_position = np.asarray(cylinder_pose.p, dtype=float)
        target_position = np.asarray(self.groove_target_pose.p, dtype=float)
        if not np.all(np.isfinite(cylinder_position)):
            raise RuntimeError("cylinder pose contains NaN or infinity")

        metrics = self.success_metrics()
        contact_points = len(
            self.get_gripper_actor_contact_position("panthera_cylinder")
        )
        measured = self._actual_robot_state()
        gripper_opening = float(measured["gripper_qpos"])
        reliably_grasped = bool(contact_points >= 2 and gripper_opening < 0.85)
        xy_error = float(np.linalg.norm(cylinder_position[:2] - target_position[:2]))
        valid_release = bool(
            not reliably_grasped
            and contact_points == 0
            and bool(metrics["gripper_open"])
            and xy_error <= 0.030
            and float(metrics["height_error_m"]) <= 0.080
            and float(metrics["axis_error_deg"]) <= 10.0
        )

        workspace = self.realized_geometry.get("workspace", {})
        base_xy = np.asarray(
            [
                workspace.get("robot_base_x_m", 0.0),
                workspace.get("robot_base_y_m", -0.35),
            ],
            dtype=float,
        )
        maximum_radius = float(workspace.get("maximum_radius_m", 0.46))
        radial_distance = float(np.linalg.norm(cylinder_position[:2] - base_xy))
        hard_failure = bool(
            cylinder_position[2] < TABLE_HEIGHT_M - 0.050
            or cylinder_position[2] > TABLE_HEIGHT_M + 1.0
            or radial_distance > maximum_radius + 0.120
        )

        if update_settle:
            current_step = int(self.simulation_step_count)
            elapsed = max(0, current_step - self._rl_last_effect_physics_step)
            self._rl_last_effect_physics_step = current_step
            if valid_release and not hard_failure:
                self._rl_release_elapsed_physics_steps += elapsed
            else:
                self._rl_release_elapsed_physics_steps = 0
                self._rl_stable_success_checks = 0
            instant_success = self._metrics_pass(metrics)
            if (
                instant_success
                and self._rl_release_elapsed_physics_steps
                >= SETTLE_SIMULATION_STEPS
            ):
                self._rl_stable_success_checks += 1
            elif not instant_success:
                self._rl_stable_success_checks = 0
            if self._rl_stable_success_checks >= 3:
                self._rl_success_latched = True

        ee_position = np.asarray(self.robot.get_left_ee_pose()[:3], dtype=float)
        return ReleaseEffectState(
            tcp_to_object_m=float(np.linalg.norm(ee_position - cylinder_position)),
            target_xy_error_m=xy_error,
            target_height_error_m=float(metrics["height_error_m"]),
            target_axis_error_deg=float(metrics["axis_error_deg"]),
            object_linear_speed_mps=float(metrics["linear_speed_mps"]),
            object_angular_speed_radps=float(metrics["angular_speed_radps"]),
            grasped=reliably_grasped,
            released_in_valid_volume=valid_release,
            success=bool(self._rl_success_latched),
            hard_failure=hard_failure,
        )

    def _top_down_grasp(self, arm: ArmTag, contact_point_id: int):
        """Approach and descend while retaining the reachable top-down attitude."""
        point = np.asarray(
            self.cylinder.get_contact_point(contact_point_id, "pose").p,
            dtype=float,
        )
        quaternion = self.grasp_quaternion_wxyz
        rotation = t3d.quaternions.quat2mat(quaternion)
        pregrasp = (
            point + rotation @ np.array([-0.15, 0.0, 0.0])
        ).tolist() + quaternion
        grasp = (
            point + rotation @ np.array([-0.12, 0.0, 0.0])
        ).tolist() + quaternion
        return arm, [
            Action(arm, "move", target_pose=pregrasp),
            Action(arm, "move", target_pose=grasp),
            Action(arm, "close", target_gripper_pos=CYLINDER_GRASP_OPENING),
        ]

    def _sample_task_scene(self, rng: np.random.Generator) -> dict:
        """Sample the legacy upright scene; subclasses may replace the contract."""
        def jitter(name: str) -> float:
            bound = float(self.task_randomization.get(name, 0.0))
            if bound < 0.0:
                raise ValueError(
                    f"task randomization bound must be nonnegative: {name}"
                )
            return float(rng.uniform(-bound, bound)) if bound else 0.0

        target_x = float(
            self.task_randomization.get("groove_x_base_m", -0.25)
        ) + jitter("groove_x_m")
        target_y = float(
            self.task_randomization.get("groove_y_base_m", -0.16)
        ) + jitter("groove_y_m")
        initial_x = float(
            self.task_randomization.get("cylinder_x_base_m", -0.25)
        ) + jitter("cylinder_x_m")
        initial_y = float(
            self.task_randomization.get("cylinder_y_base_m", -0.02)
        ) + jitter("cylinder_y_m")
        return {
            "cylinder_initial_xy_m": [initial_x, initial_y],
            "groove_target_xy_m": [target_x, target_y],
            "cylinder_posture": "upright",
            "cylinder_angle_rad": 0.0,
            "cylinder_center_z_m": (
                TABLE_HEIGHT_M + CYLINDER_HALF_HEIGHT_M + 0.002
            ),
            "cylinder_quaternion_wxyz": UPRIGHT_CYLINDER_QUATERNION_WXYZ,
            "cylinder_axis_world": [0.0, 0.0, 1.0],
        }

    def _configure_grasp_for_scene(self, scene_sample: dict):
        """Keep the reviewed top-down grasp for the legacy upright task."""
        quaternion = np.asarray(TOP_DOWN_GRASP_QUATERNION_WXYZ, dtype=float)
        self.grasp_quaternion_wxyz = (quaternion / np.linalg.norm(quaternion)).tolist()
        rotation = t3d.quaternions.quat2mat(self.grasp_quaternion_wxyz)
        self.grasp_approach_axis_world = (
            rotation @ np.array([1.0, 0.0, 0.0])
        ).tolist()
        self.finger_closing_axis_world = (
            rotation @ np.array([0.0, 1.0, 0.0])
        ).tolist()

    def load_actors(self):
        """Create an upright cylinder and a shallow yellow block/socket."""
        rng = np.random.default_rng(self.episode_seed)
        self.terminal_insertion_assist_m = float(
            os.environ.get("PANTHERA_TERMINAL_INSERTION_ASSIST_M", "0")
        )
        self.terminal_target_assist = (
            os.environ.get("PANTHERA_TERMINAL_TARGET_ASSIST", "0") == "1"
        )
        self.terminal_target_assist_trigger = os.environ.get(
            "PANTHERA_TERMINAL_TARGET_ASSIST_TRIGGER", "height"
        )
        if not math.isfinite(self.terminal_insertion_assist_m):
            raise ValueError("terminal insertion assist must be finite")
        if not 0.0 <= self.terminal_insertion_assist_m <= 0.080:
            raise ValueError("terminal insertion assist must be in [0, 0.080] m")
        if self.terminal_target_assist_trigger not in {"height", "release"}:
            raise ValueError(
                "terminal target assist trigger must be 'height' or 'release'"
            )
        self._policy_grasp_seen = False
        self._grasp_proxy_consecutive_control_steps = 0
        self._terminal_insertion_assist_attempted = False
        self._terminal_target_assist_active = False

        scene_sample = self._sample_task_scene(rng)
        initial_x, initial_y = scene_sample["cylinder_initial_xy_m"]
        target_x, target_y = scene_sample["groove_target_xy_m"]
        self.realized_geometry = scene_sample
        self.cylinder_posture = str(scene_sample["cylinder_posture"])
        self.initial_cylinder_axis_world = list(scene_sample["cylinder_axis_world"])
        self._configure_grasp_for_scene(scene_sample)

        cylinder_pose = sapien.Pose(
            [
                initial_x,
                initial_y,
                float(scene_sample["cylinder_center_z_m"]),
            ],
            scene_sample["cylinder_quaternion_wxyz"],
        )
        cylinder_entity = _create_upright_cylinder(self, cylinder_pose)
        self.cylinder = Actor(
            cylinder_entity,
            _vertical_cylinder_actor_data(),
            mass=CYLINDER_MASS_KG,
        )

        socket_color = tuple(SOCKET_BASE_COLOR[:3])
        outer_half_width = (
            SOCKET_INNER_HALF_WIDTH_M + SOCKET_WALL_HALF_WIDTH_M
        )
        floor_center_z = TABLE_HEIGHT_M + SOCKET_FLOOR_HALF_HEIGHT_M
        floor_top_z = TABLE_HEIGHT_M + 2.0 * SOCKET_FLOOR_HALF_HEIGHT_M
        wall_center_z = floor_top_z + SOCKET_WALL_HALF_HEIGHT_M
        wall_offset = SOCKET_INNER_HALF_WIDTH_M + SOCKET_WALL_HALF_WIDTH_M

        self.groove_base = create_box(
            self,
            pose=sapien.Pose([target_x, target_y, floor_center_z]),
            half_size=(
                outer_half_width,
                outer_half_width,
                SOCKET_FLOOR_HALF_HEIGHT_M,
            ),
            color=socket_color,
            is_static=True,
            name="panthera_yellow_socket_floor",
        )
        self.socket_walls = [
            create_box(
                self,
                pose=sapien.Pose(
                    [target_x, target_y - wall_offset, wall_center_z]
                ),
                half_size=(
                    outer_half_width,
                    SOCKET_WALL_HALF_WIDTH_M,
                    SOCKET_WALL_HALF_HEIGHT_M,
                ),
                color=socket_color,
                is_static=True,
                name="panthera_yellow_socket_near_wall",
            ),
            create_box(
                self,
                pose=sapien.Pose(
                    [target_x, target_y + wall_offset, wall_center_z]
                ),
                half_size=(
                    outer_half_width,
                    SOCKET_WALL_HALF_WIDTH_M,
                    SOCKET_WALL_HALF_HEIGHT_M,
                ),
                color=socket_color,
                is_static=True,
                name="panthera_yellow_socket_far_wall",
            ),
            create_box(
                self,
                pose=sapien.Pose(
                    [target_x - wall_offset, target_y, wall_center_z]
                ),
                half_size=(
                    SOCKET_WALL_HALF_WIDTH_M,
                    SOCKET_INNER_HALF_WIDTH_M,
                    SOCKET_WALL_HALF_HEIGHT_M,
                ),
                color=socket_color,
                is_static=True,
                name="panthera_yellow_socket_left_wall",
            ),
            create_box(
                self,
                pose=sapien.Pose(
                    [target_x + wall_offset, target_y, wall_center_z]
                ),
                half_size=(
                    SOCKET_WALL_HALF_WIDTH_M,
                    SOCKET_INNER_HALF_WIDTH_M,
                    SOCKET_WALL_HALF_HEIGHT_M,
                ),
                color=socket_color,
                is_static=True,
                name="panthera_yellow_socket_right_wall",
            ),
        ]
        self.socket_floor_top_z = floor_top_z
        self.socket_top_z = floor_top_z + 2.0 * SOCKET_WALL_HALF_HEIGHT_M
        self.groove_target_pose = sapien.Pose(
            [
                target_x,
                target_y,
                floor_top_z + CYLINDER_HALF_HEIGHT_M,
            ],
            UPRIGHT_CYLINDER_QUATERNION_WXYZ,
        )
        self.oracle_frames = []
        self.oracle_telemetry = []
        self._terminal_success = None
        self.add_prohibit_area(self.cylinder, padding=0.05)
        self.add_prohibit_area(self.groove_base, padding=0.05)

    def _finalize_episode_info(self):
        """Write the replayable one-arm dataset contract and latch success."""
        self.info["info"] = {"{A}": "cylinder", "{B}": "yellow socket"}
        self.info["panthera_episode"] = {
            "schema_version": int(getattr(self, "task_schema_version", TASK_SCHEMA_VERSION)),
            "scene_profile": str(
                getattr(self, "scene_profile", "phone_srt_vertical_socket")
            ),
            "episode_seed": self.episode_seed,
            "robot_source_commit": ROBOT_SOURCE_COMMIT,
            "robotwin_source_commit": ROBOTWIN_SOURCE_COMMIT,
            "physics_timestep_s": float(self.scene.get_timestep()),
            "robot_count": 1,
            "action_dimension": 7,
            "action_order": ["joint1..joint6", "gripper"],
            "realized_geometry": self.realized_geometry,
            "task_randomization": self.task_randomization,
            "domain_randomization": self.domain_randomization,
            "physics_parameters": self.physics_parameters,
            "grasp_quaternion_wxyz": self.grasp_quaternion_wxyz,
            "grasp_approach_axis_world": self.grasp_approach_axis_world,
            "finger_closing_axis_world": self.finger_closing_axis_world,
            "initial_cylinder_axis_world": self.initial_cylinder_axis_world,
            "cylinder_axis_world": self.initial_cylinder_axis_world,
            "cylinder_posture": self.cylinder_posture,
            "cylinder_radius_m": CYLINDER_RADIUS_M,
            "cylinder_height_m": 2.0 * CYLINDER_HALF_HEIGHT_M,
            "socket_inner_half_width_m": SOCKET_INNER_HALF_WIDTH_M,
            "socket_floor_top_z_m": self.socket_floor_top_z,
            "socket_top_z_m": self.socket_top_z,
            "sample_period_physics_steps": int(self.save_freq),
            "attach_on_grasp": False,
            "final_settle_simulation_steps": self.final_settle_simulation_steps,
        }
        self._terminal_success = self._metrics_pass(self.success_metrics())

    def play_once(self):
        """Top-down grasp, lift, align, insert while held, release and retract."""
        arm = ArmTag("left")
        if not self._move_arm(
            self.close_gripper(arm, pos=PREGRASP_OPENING), "preclose"
        ):
            return self.info
        if not self._move_arm(self._top_down_grasp(arm, 0), "top_down_grasped"):
            return self.info
        self._advance_physics(25)
        self.capture_oracle_stage("grasp_settled")
        if len(
            self.get_gripper_actor_contact_position("panthera_cylinder")
        ) < 2:
            self.capture_oracle_stage("grasp_contact_rejected")
            return self.info

        current = np.asarray(self.cylinder.get_pose().p, dtype=float)
        carry_height = max(
            float(current[2] + 0.0875),
            float(self.groove_target_pose.p[2] + 0.0275),
        )
        lift_waypoints = [
            current + np.array([0.0, 0.0, 0.035]),
            current + np.array([0.0, -0.015, 0.070]),
            current + np.array([0.0, -0.030, 0.070]),
            current + np.array([0.0, -0.030, 0.0875]),
        ]
        for index, waypoint in enumerate(lift_waypoints, start=1):
            if not self._carry_object_center_to(
                waypoint,
                f"lifted_{index}",
                max_segment_m=0.035,
            ):
                return self.info
        socket_clearance_height = float(self.groove_target_pose.p[2] + 0.045)
        if not self._carry_object_center_to(
            np.array(
                [
                    self.groove_target_pose.p[0],
                    self.groove_target_pose.p[1] + 0.085,
                    carry_height,
                ]
            ),
            "staged_before_socket",
            max_segment_m=0.035,
        ):
            return self.info
        if not self._carry_object_center_to(
            np.array(
                [
                    self.groove_target_pose.p[0],
                    self.groove_target_pose.p[1] + 0.085,
                    socket_clearance_height,
                ]
            ),
            "raised_for_socket",
            max_segment_m=0.035,
        ):
            return self.info
        if not self._carry_object_center_to(
            np.array(
                [
                    self.groove_target_pose.p[0],
                    self.groove_target_pose.p[1],
                    socket_clearance_height,
                ]
            ),
            "aligned_above_socket",
            max_segment_m=0.035,
        ):
            return self.info
        if not self._carry_object_center_to(
            np.asarray(self.groove_target_pose.p, dtype=float)
            + np.array([0.0, 0.0, 0.025]),
            "preinserted",
            tolerance_m=0.010,
            max_segment_m=0.035,
        ):
            return self.info
        if not self._carry_object_center_to(
            np.asarray(self.groove_target_pose.p, dtype=float),
            "inserted_while_grasped",
            tolerance_m=0.010,
            max_segment_m=0.035,
        ):
            return self.info
        if not self._move_arm(self.open_gripper(arm), "released_in_socket"):
            return self.info
        self._advance_physics(25)
        self.capture_oracle_stage("release_settled")
        withdrawals = [
            (0.0, 0.0, 0.03),
        ]
        for index, displacement in enumerate(withdrawals, start=1):
            if not self._move_arm(
                self.move_by_displacement(arm, *displacement),
                f"gripper_withdrawn_{index}",
            ):
                return self.info
        self._advance_until_inserted_stable()
        self.capture_oracle_stage("inserted_and_settled")

        self._finalize_episode_info()
        return self.info

    def gen_sparse_reward_data(self, chunk_actions, action_type="qpos"):
        """Execute one policy chunk and optionally persist closed-loop telemetry.

        Setting ``PANTHERA_EVAL_TRACE_DIR`` enables one JSONL file per simulator
        process. The trace is event-driven: one row is written immediately after
        each action chunk, with no wall-clock delay in the rollout.
        """
        actions = np.asarray(chunk_actions, dtype=float)
        before_state = np.asarray(
            self._actual_robot_state()["vector"], dtype=float
        )
        before_cylinder = np.asarray(self.cylinder.get_pose().p, dtype=float)
        if GRASP_PROXY_MIN_OPENING <= before_state[6] < GRASP_PROXY_MAX_OPENING:
            self._grasp_proxy_consecutive_control_steps += len(actions)
        elif not self._policy_grasp_seen:
            self._grasp_proxy_consecutive_control_steps = 0
        if (
            self._grasp_proxy_consecutive_control_steps
            >= GRASP_PROXY_REQUIRED_CONTROL_STEPS
        ):
            self._policy_grasp_seen = True
        release_requested = bool(
            self._policy_grasp_seen and np.any(actions[:, 6] >= 0.80)
        )
        current_ee_pose = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        current_ee_position = current_ee_pose[:3]
        current_ee_quaternion = current_ee_pose[3:]
        target_ee_position = np.asarray(
            self.groove_target_pose.p, dtype=float
        ) + np.array(
            [
                0.0,
                -NOMINAL_CYLINDER_CENTER_FROM_EE_Y_M,
                NOMINAL_GRIPPER_TO_CYLINDER_CENTER_M,
            ]
        )
        target_entry_error = current_ee_position - target_ee_position
        calibrated_target_entry = bool(
            self._policy_grasp_seen
            and np.linalg.norm(target_entry_error[:2]) <= 0.030
            and -0.020 <= target_entry_error[2] <= 0.080
        )
        grasp_handoff_ready = bool(
            self._policy_grasp_seen
            and GRASP_PROXY_MIN_OPENING
            <= before_state[6]
            < GRASP_PROXY_MAX_OPENING
            and current_ee_position[2] >= target_ee_position[2]
        )
        if self.terminal_target_assist_trigger == "release":
            target_assist_ready = bool(
                release_requested
                and calibrated_target_entry
                and target_entry_error[2] > TERMINAL_ASSIST_RELEASE_Z_MIN_M
            )
        else:
            target_assist_ready = grasp_handoff_ready
        assist_attempted = False
        assist_succeeded = False
        assist_stage_results = {}
        assist_stage_diagnostics = {}
        if (
            (self.terminal_insertion_assist_m > 0.0 or self.terminal_target_assist)
            and (
                target_assist_ready
                if self.terminal_target_assist
                else release_requested
            )
            and not self._terminal_insertion_assist_attempted
        ):
            # This is a deployable terminal skill, not an oracle object-pose
            # correction: it reads only the robot's measured end-effector pose.
            # The staged path takes over after the proprioceptive grasp proxy
            # and after the policy has lifted the tool to the calibrated
            # insertion height.  That keeps the learned grasp/lift phase while
            # avoiding its longer carry, where the object accumulated tilt.
            self._terminal_insertion_assist_attempted = True
            assist_attempted = True
            assist_stage_diagnostics["before_handoff_settle"] = (
                self._terminal_assist_diagnostic_snapshot()
            )
            if self.terminal_target_assist:
                # The learned policy may reach the handoff height while the
                # grasped object and arm are still moving. Hold the last motor
                # targets and advance physics before starting the planned
                # route. This uses no object-state decision or wall-clock
                # sleep; object telemetry below is diagnostic only.
                self._advance_physics(TERMINAL_ASSIST_HANDOFF_SETTLE_STEPS)
                assist_stage_results["handoff_settle_steps"] = (
                    TERMINAL_ASSIST_HANDOFF_SETTLE_STEPS
                )
                current_ee_pose = np.asarray(
                    self.robot.get_left_ee_pose(), dtype=float
                )
                current_ee_position = current_ee_pose[:3]
                current_ee_quaternion = current_ee_pose[3:]
                assist_stage_diagnostics["after_handoff_settle"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                # The socket pose represents a calibrated workstation target.
                # The target does not depend on the simulated cylinder pose.
                target_hover_z = float(
                    target_ee_position[2] + TERMINAL_ASSIST_HOVER_OFFSET_M
                )
                departure_ee_position = current_ee_position.copy()
                departure_ee_position[2] = min(
                    target_hover_z,
                    float(
                        current_ee_position[2]
                        + TERMINAL_ASSIST_DEPARTURE_LIFT_M
                    ),
                )
                lifted, lift_segments = self._move_terminal_ee_segmented(
                    departure_ee_position,
                    target_quaternion_wxyz=current_ee_quaternion,
                    max_segment_m=TERMINAL_ASSIST_ALIGN_SEGMENT_M,
                    settle_steps=TERMINAL_ASSIST_ALIGN_SETTLE_STEPS,
                )
                assist_stage_results["lifted_from_source"] = lifted
                assist_stage_results["lift_segments"] = lift_segments
                assist_stage_diagnostics["after_lift"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                if self.terminal_target_assist_trigger == "release":
                    # The learned policy has already carried the object into
                    # the calibrated socket neighborhood. Do not retreat to
                    # the long approach waypoint; only establish clearance.
                    approach_ee_position = departure_ee_position.copy()
                    approached = lifted
                    approach_segments = 0
                else:
                    approach_ee_position = target_ee_position.copy()
                    approach_ee_position[1] += (
                        TERMINAL_ASSIST_APPROACH_Y_OFFSET_M
                    )
                    approach_ee_position[2] = departure_ee_position[2]
                    approached = False
                    approach_segments = 0
                    if lifted:
                        approached, approach_segments = (
                            self._move_terminal_ee_segmented(
                                approach_ee_position,
                                target_quaternion_wxyz=current_ee_quaternion,
                                max_segment_m=TERMINAL_ASSIST_ALIGN_SEGMENT_M,
                                settle_steps=TERMINAL_ASSIST_ALIGN_SETTLE_STEPS,
                            )
                        )
                assist_stage_results["reached_socket_approach"] = approached
                assist_stage_results["approach_segments"] = approach_segments
                assist_stage_diagnostics["after_approach"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                clearance_ee_position = approach_ee_position.copy()
                clearance_ee_position[2] = target_hover_z
                raised = False
                clearance_segments = 0
                if approached:
                    raised, clearance_segments = (
                        self._move_terminal_ee_segmented(
                            clearance_ee_position,
                            target_quaternion_wxyz=current_ee_quaternion,
                            max_segment_m=TERMINAL_ASSIST_ALIGN_SEGMENT_M,
                            settle_steps=TERMINAL_ASSIST_ALIGN_SETTLE_STEPS,
                        )
                    )
                assist_stage_results["raised_to_clearance"] = raised
                assist_stage_results["clearance_segments"] = clearance_segments
                assist_stage_diagnostics["after_clearance"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                oriented = False
                if raised:
                    # Establish the reviewed insertion attitude while the
                    # cylinder is still away from the socket walls. Performing
                    # this rotation after horizontal alignment caused rim
                    # contact and grasp slip on randomized seeds.
                    orient_action = self.move_to_pose(
                        ArmTag("left"),
                        clearance_ee_position.tolist()
                        + TOP_DOWN_GRASP_QUATERNION_WXYZ,
                    )
                    oriented = bool(
                        self.move(orient_action, save_freq=None)
                        and self.plan_success
                    )
                    if oriented:
                        self._advance_physics(
                            TERMINAL_ASSIST_ALIGN_SETTLE_STEPS
                        )
                assist_stage_results["oriented_at_clearance"] = oriented
                assist_stage_diagnostics["after_orientation"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                align_ee_position = target_ee_position.copy()
                align_ee_position[2] = target_hover_z
                aligned = False
                align_segments = 0
                if oriented:
                    aligned, align_segments = self._move_terminal_ee_segmented(
                        align_ee_position,
                        target_quaternion_wxyz=(
                            TOP_DOWN_GRASP_QUATERNION_WXYZ
                        ),
                        max_segment_m=TERMINAL_ASSIST_ALIGN_SEGMENT_M,
                        settle_steps=TERMINAL_ASSIST_ALIGN_SETTLE_STEPS,
                    )
                assist_stage_results["aligned_above_socket"] = aligned
                assist_stage_results["align_segments"] = align_segments
                assist_stage_diagnostics["after_align"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                inserted = False
                released = False
                if aligned:
                    inserted, insert_segments = self._move_terminal_ee_segmented(
                        target_ee_position,
                        target_quaternion_wxyz=(
                            TOP_DOWN_GRASP_QUATERNION_WXYZ
                        ),
                        max_segment_m=TERMINAL_ASSIST_INSERT_SEGMENT_M,
                        settle_steps=TERMINAL_ASSIST_INSERT_SETTLE_STEPS,
                    )
                else:
                    insert_segments = 0
                assist_stage_results["inserted"] = inserted
                assist_stage_results["insert_segments"] = insert_segments
                assist_stage_diagnostics["after_insert"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                if inserted:
                    # Let contact dynamics settle while the cylinder is still
                    # supported. This advances simulation physics directly;
                    # it is not a wall-clock sleep and does not break the
                    # continuous episode trace.
                    self._advance_physics(
                        TERMINAL_ASSIST_PRE_RELEASE_SETTLE_STEPS
                    )
                    assist_stage_results["pre_release_settle_steps"] = (
                        TERMINAL_ASSIST_PRE_RELEASE_SETTLE_STEPS
                    )
                    assist_stage_diagnostics["before_release"] = (
                        self._terminal_assist_diagnostic_snapshot()
                    )
                    released = bool(
                        self.move(
                            self.open_gripper(ArmTag("left")),
                            save_freq=None,
                        )
                    )
                assist_stage_results["released"] = released
                assist_stage_diagnostics["after_release"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                if released:
                    self._advance_physics(
                        TERMINAL_ASSIST_POST_RELEASE_SETTLE_STEPS
                    )
                    assist_stage_results["post_release_settle_steps"] = (
                        TERMINAL_ASSIST_POST_RELEASE_SETTLE_STEPS
                    )
                    assist_stage_diagnostics["after_release_settle"] = (
                        self._terminal_assist_diagnostic_snapshot()
                    )
                withdrawn = False
                withdraw_segments = 0
                if released:
                    withdraw_target = np.asarray(
                        self.robot.get_left_ee_pose()[:3], dtype=float
                    ) + np.array([0.0, 0.0, 0.030])
                    withdrawn, withdraw_segments = (
                        self._move_terminal_ee_segmented(
                            withdraw_target,
                            target_quaternion_wxyz=(
                                TOP_DOWN_GRASP_QUATERNION_WXYZ
                            ),
                            max_segment_m=TERMINAL_ASSIST_ALIGN_SEGMENT_M,
                            settle_steps=TERMINAL_ASSIST_ALIGN_SETTLE_STEPS,
                        )
                    )
                assist_stage_results["withdrawn_after_release"] = withdrawn
                assist_stage_results["withdraw_segments"] = withdraw_segments
                assist_stage_diagnostics["after_withdraw"] = (
                    self._terminal_assist_diagnostic_snapshot()
                )
                assist_succeeded = (
                    lifted
                    and approached
                    and raised
                    and aligned
                    and oriented
                    and inserted
                    and released
                    and withdrawn
                )
            else:
                assist_action = self.move_by_displacement(
                    ArmTag("left"),
                    z=-self.terminal_insertion_assist_m,
                    quat=TOP_DOWN_GRASP_QUATERNION_WXYZ,
                )
                assist_succeeded = bool(self.move(assist_action, save_freq=None))
                assist_stage_results["relative_down"] = assist_succeeded
            self._terminal_target_assist_active = bool(
                self.terminal_target_assist and assist_succeeded
            )
        if self._terminal_target_assist_active:
            # Keep the calibrated insertion pose while retaining the VLA's
            # gripper channel. This prevents its learned post-release retraction
            # from interrupting settling inside the socket.
            hold_state = np.asarray(
                self._actual_robot_state()["vector"], dtype=float
            )
            actions = actions.copy()
            actions[:, :6] = hold_state[:6]
            actions[:, 6] = 1.0
        result = super().gen_sparse_reward_data(actions, action_type=action_type)

        trace_dir = os.environ.get("PANTHERA_EVAL_TRACE_DIR")
        if trace_dir:
            after_state = np.asarray(
                self._actual_robot_state()["vector"], dtype=float
            )
            after_cylinder = np.asarray(self.cylinder.get_pose().p, dtype=float)
            trace_index = int(getattr(self, "_policy_trace_index", 0))
            self._policy_trace_index = trace_index + 1
            record = {
                "chunk_index": trace_index,
                "episode_seed": int(self.episode_seed),
                "before_state": before_state.tolist(),
                "action_first": actions[0].tolist(),
                "action_last": actions[-1].tolist(),
                "action_min": actions.min(axis=0).tolist(),
                "action_max": actions.max(axis=0).tolist(),
                "after_state": after_state.tolist(),
                "state_delta": (after_state - before_state).tolist(),
                "before_cylinder_position_m": before_cylinder.tolist(),
                "after_cylinder_position_m": after_cylinder.tolist(),
                "cylinder_displacement_m": float(
                    np.linalg.norm(after_cylinder - before_cylinder)
                ),
                "success_metrics": self.success_metrics(),
                "reward": np.asarray(result[0]).tolist(),
                "termination": np.asarray(result[1]).tolist(),
                "truncation": np.asarray(result[2]).tolist(),
                "terminal_insertion_assist": {
                    "configured_distance_m": self.terminal_insertion_assist_m,
                    "current_ee_position_m": current_ee_position.tolist(),
                    "current_ee_quaternion_wxyz": (
                        current_ee_quaternion.tolist()
                    ),
                    "target_ee_position_m": target_ee_position.tolist(),
                    "target_entry_error_m": target_entry_error.tolist(),
                    "grasp_proxy_consecutive_control_steps": (
                        self._grasp_proxy_consecutive_control_steps
                    ),
                    "policy_grasp_seen": self._policy_grasp_seen,
                    "mode": (
                        "calibrated_socket_target"
                        if self.terminal_target_assist
                        else "relative_down"
                    ),
                    "target_assist_trigger": (
                        self.terminal_target_assist_trigger
                    ),
                    "release_z_min_m": TERMINAL_ASSIST_RELEASE_Z_MIN_M,
                    "release_requested": release_requested,
                    "calibrated_target_entry": calibrated_target_entry,
                    "grasp_handoff_ready": grasp_handoff_ready,
                    "attempted": assist_attempted,
                    "succeeded": assist_succeeded,
                    "active": self._terminal_target_assist_active,
                    "stage_results": assist_stage_results,
                    "stage_diagnostics": assist_stage_diagnostics,
                },
            }
            destination = Path(trace_dir)
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / f"policy-env-{os.getpid()}.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return result

    def _terminal_assist_diagnostic_snapshot(self) -> dict:
        """Capture object and robot telemetry without influencing control."""
        cylinder_pose = self.cylinder.get_pose()
        ee_pose = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        state = np.asarray(self._actual_robot_state()["vector"], dtype=float)
        return {
            "cylinder_position_m": np.asarray(
                cylinder_pose.p, dtype=float
            ).tolist(),
            "cylinder_quaternion_wxyz": np.asarray(
                cylinder_pose.q, dtype=float
            ).tolist(),
            "ee_position_m": ee_pose[:3].tolist(),
            "ee_quaternion_wxyz": ee_pose[3:].tolist(),
            "gripper_opening": float(state[6]),
            "success_metrics": self.success_metrics(),
        }

    def _move_terminal_ee_segmented(
        self,
        target_position: np.ndarray,
        *,
        target_quaternion_wxyz: list[float] | np.ndarray,
        max_segment_m: float,
        settle_steps: int,
    ) -> tuple[bool, int]:
        """Move the measured EE to a calibrated target in bounded segments.

        This helper deliberately reads only robot proprioception and the fixed
        workstation target. Short Cartesian segments reduce grasp slip and rim
        impact without consulting the simulated cylinder pose.
        """
        if not math.isfinite(max_segment_m) or max_segment_m <= 0.0:
            raise ValueError("terminal assist max segment must be positive")
        if settle_steps < 0:
            raise ValueError("terminal assist settle steps must be nonnegative")
        target = np.asarray(target_position, dtype=float)
        current = np.asarray(self.robot.get_left_ee_pose()[:3], dtype=float)
        distance = float(np.linalg.norm(target - current))
        segment_count = max(
            1,
            int(math.ceil(distance / max_segment_m)),
        )
        for segment_index in range(segment_count):
            current = np.asarray(self.robot.get_left_ee_pose()[:3], dtype=float)
            remaining_segments = segment_count - segment_index
            waypoint = current + (target - current) / float(remaining_segments)
            action = self.move_to_pose(
                ArmTag("left"),
                waypoint.tolist()
                + np.asarray(target_quaternion_wxyz, dtype=float).tolist(),
            )
            if not self.move(action, save_freq=None) or not self.plan_success:
                return False, segment_index + 1
            self._advance_physics(settle_steps)
        return True, segment_count

    def _advance_until_inserted_stable(self) -> bool:
        """Advance simulation until three consecutive stable checks or 5 s."""
        check_interval = 25
        minimum_steps = SETTLE_SIMULATION_STEPS
        maximum_steps = 5 * SETTLE_SIMULATION_STEPS
        stable_checks = 0
        elapsed_steps = 0
        while elapsed_steps < maximum_steps:
            self._advance_physics(check_interval)
            elapsed_steps += check_interval
            if elapsed_steps < minimum_steps:
                continue
            if self._metrics_pass(self.success_metrics()):
                stable_checks += 1
                if stable_checks >= 3:
                    self.final_settle_simulation_steps = elapsed_steps
                    return True
            else:
                stable_checks = 0
        self.final_settle_simulation_steps = elapsed_steps
        return False

    def success_metrics(self) -> dict[str, float | bool]:
        """Measure horizontal centering, insertion depth and verticality."""
        pose = self.cylinder.get_pose()
        position_error = np.asarray(pose.p) - np.asarray(
            self.groove_target_pose.p
        )
        cylinder_axis = t3d.quaternions.quat2mat(pose.q)[:, 0]
        axis_dot = float(
            np.clip(abs(np.dot(cylinder_axis, [0.0, 0.0, 1.0])), 0.0, 1.0)
        )
        axis_error_deg = math.degrees(math.acos(axis_dot))
        rigid = next(
            component
            for component in self.cylinder.actor.get_components()
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
        )
        cylinder_bottom_z = float(pose.p[2] - CYLINDER_HALF_HEIGHT_M)
        insertion_depth = float(self.socket_top_z - cylinder_bottom_z)
        gripper_contact_impulse = 0.0
        for contact in self.scene.get_contacts():
            names = {body.entity.name for body in contact.bodies}
            if "panthera_cylinder" not in names:
                continue
            if not names.intersection({"L_finger", "R_finger"}):
                continue
            gripper_contact_impulse += sum(
                float(np.linalg.norm(point.impulse)) for point in contact.points
            )
        return {
            "x_error_m": float(abs(position_error[0])),
            "y_error_m": float(abs(position_error[1])),
            "height_error_m": float(abs(position_error[2])),
            "axis_error_deg": float(axis_error_deg),
            "insertion_depth_m": insertion_depth,
            "linear_speed_mps": float(np.linalg.norm(rigid.get_linear_velocity())),
            "angular_speed_radps": float(
                np.linalg.norm(rigid.get_angular_velocity())
            ),
            "gripper_open": bool(self.is_left_gripper_open()),
            "gripper_contact_impulse_ns": gripper_contact_impulse,
        }

    @staticmethod
    def _metrics_pass(metrics: dict[str, float | bool]) -> bool:
        return bool(
            metrics["x_error_m"] <= 0.010
            and metrics["y_error_m"] <= 0.010
            and metrics["height_error_m"] <= 0.012
            and metrics["axis_error_deg"] <= 5.0
            and metrics["insertion_depth_m"] >= 0.030
            and metrics["linear_speed_mps"] <= 0.025
            and metrics["angular_speed_radps"] <= 0.35
            and metrics["gripper_open"]
            and metrics["gripper_contact_impulse_ns"] <= 1.0e-6
        )

    def get_info(self):
        return {"{A}": "vertical cylinder", "{B}": "yellow socket"}
