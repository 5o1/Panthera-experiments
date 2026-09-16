"""Single-Panthera cylinder-in-groove task for the sim-to-real path.

One six-axis Panthera and its parallel gripper grasp a horizontal cylinder at
the center, carry it to a straight U-shaped groove, release it, and retract.
The external observation/action contract is seven dimensional: six joints plus
one normalized gripper command.
"""

import math
import os

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import Action, ArmTag, Actor, create_box


TABLE_HEIGHT_M = 0.74
CYLINDER_RADIUS_M = 0.0275
CYLINDER_HALF_LENGTH_M = 0.12
# The 36 mm inner half-width accepts the declared <=6 degree cylinder-axis
# tolerance across the 50 mm guide half-length, including collision margins.
GROOVE_INNER_HALF_WIDTH_M = 0.036
GROOVE_RAIL_HALF_WIDTH_M = 0.012
GROOVE_RAIL_HALF_HEIGHT_M = 0.020
GROOVE_RAIL_HALF_LENGTH_M = 0.075
GROOVE_BASE_HALF_HEIGHT_M = 0.005
SETTLE_SIMULATION_STEPS = 250
LEFT_TOP_DOWN_QUATERNION_WXYZ = [-0.5, 0.5, -0.5, -0.5]
PANTHERA_DIAMETER_GRASP_QUATERNION_WXYZ = [
    math.sqrt(0.5),
    0.0,
    math.sqrt(0.5),
    0.0,
]
PREGRASP_OPENING = 0.90
CYLINDER_GRASP_OPENING = 0.0
TASK_SCHEMA_VERSION = 3
ROBOT_SOURCE_COMMIT = "b08633d6c5bce89baad1821fd598243a84bc3a84"
ROBOTWIN_SOURCE_COMMIT = "0008ae6800df9f75fc8de7098bacb01735fd8fd2"


def _create_grippable_cylinder(task, pose: sapien.Pose) -> sapien.Entity:
    """Create one plain rigid cylinder for a center grasp by one gripper."""
    entity = sapien.Entity()
    entity.set_name("panthera_cylinder")
    entity.set_pose(pose)

    physical_material = task.scene.create_physical_material(3.0, 2.0, 0.0)
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=CYLINDER_RADIUS_M,
            half_length=CYLINDER_HALF_LENGTH_M,
            material=physical_material,
        )
    )

    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        sapien.render.RenderShapeCylinder(
            radius=CYLINDER_RADIUS_M,
            half_length=CYLINDER_HALF_LENGTH_M,
            material=sapien.render.RenderMaterial(
                base_color=[0.95, 0.65, 0.05, 1.0]
            ),
        )
    )
    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)
    task.scene.add_entity(entity)
    return entity


def _frame(rotation: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    """Return a homogeneous contact/functional frame for RoboTwin Actor metadata."""
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result.tolist()


def _cylinder_actor_data() -> dict:
    """Describe one center grasp station and the cylinder-center target point."""
    center_grasp_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    identity = np.eye(4, dtype=float).tolist()
    return {
        "center": [0.0, 0.0, 0.0],
        "extents": [
            2.0 * CYLINDER_HALF_LENGTH_M,
            2.0 * CYLINDER_RADIUS_M,
            2.0 * CYLINDER_RADIUS_M,
        ],
        "scale": [1.0, 1.0, 1.0],
        "target_pose": [identity],
        "contact_points_pose": [
            _frame(center_grasp_rotation, np.zeros(3, dtype=float)),
        ],
        "functional_matrix": [identity],
        "orientation_point": [identity],
        "contact_points_group": [[0]],
        "contact_points_mask": [True],
        "contact_points_description": ["center grip station"],
        "target_point_description": ["cylinder center"],
    }


class place_cylinder_in_groove(Base_Task):
    """RoboTwin task with a bounded, deterministic single-arm scripted oracle."""

    def setup_scene(self, **kwargs):
        """Create the base scene, then disable the unstable OIDN post-process."""
        super().setup_scene(**kwargs)
        denoiser = os.environ.get("PANTHERA_ROBOTWIN_DENOISER", "none")
        if denoiser not in {"none", "oidn", "optix"}:
            raise ValueError(f"unsupported SAPIEN denoiser: {denoiser}")
        sapien.render.set_ray_tracing_denoiser(denoiser)

    def setup_demo(self, **kwargs):
        """Create one Panthera, the cameras, a cylinder and its target groove."""
        self.episode_seed = int(kwargs.get("seed", 0))
        self.task_randomization = kwargs.get("task_randomization", {}) or {}
        self.domain_randomization = kwargs.get("domain_randomization", {}) or {}
        configured_grasp = kwargs.get("grasp_quaternion_wxyz")
        if configured_grasp is None:
            embodiment_name = str(kwargs.get("embodiment_name", "")).lower()
            configured_grasp = (
                PANTHERA_DIAMETER_GRASP_QUATERNION_WXYZ
                if "panthera" in embodiment_name
                else LEFT_TOP_DOWN_QUATERNION_WXYZ
            )
        grasp_quaternion = np.asarray(configured_grasp, dtype=float)
        if grasp_quaternion.shape != (4,) or not np.all(np.isfinite(grasp_quaternion)):
            raise ValueError("grasp_quaternion_wxyz must contain four finite values")
        quaternion_norm = float(np.linalg.norm(grasp_quaternion))
        if quaternion_norm <= 1.0e-10:
            raise ValueError("grasp_quaternion_wxyz must be nonzero")
        self.grasp_quaternion_wxyz = (grasp_quaternion / quaternion_norm).tolist()
        grasp_rotation = t3d.quaternions.quat2mat(self.grasp_quaternion_wxyz)
        self.grasp_approach_axis_world = (grasp_rotation @ np.array([1.0, 0.0, 0.0])).tolist()
        self.finger_closing_axis_world = (grasp_rotation @ np.array([0.0, 1.0, 0.0])).tolist()
        self.physics_parameters = {
            "static_friction": float(kwargs.get("static_friction", 1.2)),
            "dynamic_friction": float(kwargs.get("dynamic_friction", 1.0)),
            "restitution": float(kwargs.get("restitution", 0.0)),
            "cylinder_mass_kg": 0.03,
        }
        super()._init_task_env_(**kwargs)
        if not self.robot.is_dual_arm:
            raise ValueError("single-Panthera task requires embodiment: [panthera]")
        if self.robot.left_entity is not self.robot.right_entity:
            raise ValueError("RoboTwin single embodiment did not alias one articulation")
        self.single_arm_mode = True

    def load_actors(self):
        """Create a horizontal dynamic cylinder and a static U-shaped target groove."""
        rng = np.random.default_rng(self.episode_seed)

        def jitter(name: str) -> float:
            bound = float(self.task_randomization.get(name, 0.0))
            if bound < 0.0:
                raise ValueError(f"task randomization bound must be nonnegative: {name}")
            return float(rng.uniform(-bound, bound)) if bound else 0.0

        target_x = jitter("groove_x_m")
        target_y = -0.16 + jitter("groove_y_m")
        initial_x = jitter("cylinder_x_m")
        initial_y = float(self.task_randomization.get("cylinder_y_base_m", -0.08)) + jitter("cylinder_y_m")
        self.realized_geometry = {
            "cylinder_initial_xy_m": [initial_x, initial_y],
            "groove_target_xy_m": [target_x, target_y],
        }

        cylinder_pose = sapien.Pose(
            [
                initial_x,
                initial_y,
                TABLE_HEIGHT_M + CYLINDER_RADIUS_M + 0.002,
            ],
            [1.0, 0.0, 0.0, 0.0],
        )
        cylinder_entity = _create_grippable_cylinder(self, cylinder_pose)
        self.cylinder = Actor(cylinder_entity, _cylinder_actor_data(), mass=0.03)

        groove_half_length = CYLINDER_HALF_LENGTH_M + 0.018
        groove_total_half_width = GROOVE_INNER_HALF_WIDTH_M + GROOVE_RAIL_HALF_WIDTH_M
        base_center_z = TABLE_HEIGHT_M + GROOVE_BASE_HALF_HEIGHT_M
        rail_center_z = TABLE_HEIGHT_M + 2.0 * GROOVE_BASE_HALF_HEIGHT_M + GROOVE_RAIL_HALF_HEIGHT_M
        rail_offset_y = GROOVE_INNER_HALF_WIDTH_M + GROOVE_RAIL_HALF_WIDTH_M
        groove_color = (0.18, 0.42, 0.82)

        self.groove_base = create_box(
            self,
            pose=sapien.Pose([target_x, target_y, base_center_z]),
            half_size=(groove_half_length, groove_total_half_width, GROOVE_BASE_HALF_HEIGHT_M),
            color=groove_color,
            is_static=True,
            name="panthera_groove_base",
        )
        self.groove_left_rail = create_box(
            self,
            pose=sapien.Pose([target_x, target_y - rail_offset_y, rail_center_z]),
            half_size=(GROOVE_RAIL_HALF_LENGTH_M, GROOVE_RAIL_HALF_WIDTH_M, GROOVE_RAIL_HALF_HEIGHT_M),
            color=groove_color,
            is_static=True,
            name="panthera_groove_left_rail",
        )
        self.groove_right_rail = create_box(
            self,
            pose=sapien.Pose([target_x, target_y + rail_offset_y, rail_center_z]),
            half_size=(GROOVE_RAIL_HALF_LENGTH_M, GROOVE_RAIL_HALF_WIDTH_M, GROOVE_RAIL_HALF_HEIGHT_M),
            color=groove_color,
            is_static=True,
            name="panthera_groove_right_rail",
        )

        actual_base_pose = self.groove_base.get_pose().p
        self.groove_target_pose = sapien.Pose(
            [actual_base_pose[0], actual_base_pose[1], actual_base_pose[2] + GROOVE_BASE_HALF_HEIGHT_M + CYLINDER_RADIUS_M],
            [1.0, 0.0, 0.0, 0.0],
        )
        self.oracle_frames: list[tuple[str, np.ndarray]] = []
        self.oracle_telemetry: list[dict[str, object]] = []
        # RoboTwin's official collector calls check_success() only after
        # close_env() has released self.robot.  Latch the terminal result while
        # the articulation is still alive so post-save validation is safe.
        self._terminal_success: bool | None = None
        self.add_prohibit_area(self.cylinder, padding=0.05)
        self.add_prohibit_area(self.groove_base, padding=0.05)

    def capture_oracle_stage(self, stage: str):
        """Capture one current head-camera frame without wall-clock sleeping."""
        observation = self.get_obs()
        frame = observation["observation"]["head_camera"]["rgb"].copy()
        self.oracle_frames.append((stage, frame))
        cylinder_pose = self.cylinder.get_pose()

        planner_collisions: dict[str, list[list[str]]] = {}
        planner_wrapper = getattr(self.robot, "left_planner", None)
        planner = getattr(planner_wrapper, "planner", None)
        if planner is not None and hasattr(planner, "check_for_self_collision"):
            def pairs(collisions):
                return [
                    [
                        collision.object_name1,
                        collision.link_name1,
                        collision.object_name2,
                        collision.link_name2,
                    ]
                    for collision in collisions
                ]

            qpos = self.robot.left_entity.get_qpos()
            planner_collisions = {
                "self": pairs(planner.check_for_self_collision(qpos)),
                "environment": pairs(planner.check_for_env_collision(qpos)),
            }

        object_contacts: dict[str, int] = {}
        object_contact_impulse_ns: dict[str, float] = {}
        for contact in self.scene.get_contacts():
            names = [body.entity.name for body in contact.bodies]
            if "panthera_cylinder" not in names:
                continue
            other = names[1] if names[0] == "panthera_cylinder" else names[0]
            object_contacts[other] = object_contacts.get(other, 0) + len(contact.points)
            object_contact_impulse_ns[other] = (
                object_contact_impulse_ns.get(other, 0.0)
                + sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
            )
        rigid_body = next(
            component
            for component in self.cylinder.actor.get_components()
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
        )
        self.oracle_telemetry.append(
            {
                "stage": stage,
                "cylinder_position": cylinder_pose.p.tolist(),
                "cylinder_quaternion_wxyz": cylinder_pose.q.tolist(),
                "tcp": list(self.robot.get_left_tcp_pose()),
                "gripper": float(self.robot.get_left_gripper_val()),
                "qpos": self.robot.left_entity.get_qpos().tolist(),
                "planner_collisions": planner_collisions,
                "object_contacts": object_contacts,
                "object_contact_impulse_ns": object_contact_impulse_ns,
                "cylinder_sleeping": bool(rigid_body.is_sleeping),
                "gripper_contact_points": len(
                    self.get_gripper_actor_contact_position("panthera_cylinder")
                ),
            }
        )

    def _actual_robot_state(self) -> dict[str, np.ndarray | float]:
        """Read one articulation as six measured joints plus finger opening."""
        entity = self.robot.left_entity
        active_joints = entity.get_active_joints()
        active_qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        qpos_by_name = {
            joint.get_name(): float(active_qpos[index])
            for index, joint in enumerate(active_joints)
        }
        arm_qpos = np.asarray(
            [qpos_by_name[joint.get_name()] for joint in self.robot.left_arm_joints],
            dtype=np.float64,
        )
        gripper_joint = self.robot.left_gripper[0][0]
        finger_qpos = qpos_by_name[gripper_joint.get_name()]
        scale = np.asarray(self.robot.left_gripper_scale, dtype=float)
        gripper_opening = float(
            np.clip((finger_qpos - scale[0]) / (scale[1] - scale[0]), 0.0, 1.0)
        )
        return {
            "arm_qpos": arm_qpos,
            "gripper_qpos": gripper_opening,
            "vector": np.concatenate((arm_qpos, [gripper_opening])),
        }

    def get_obs(self):
        """Expose a measured 7-D single-arm state and exact simulation clock."""
        observation = super().get_obs()
        joint_action = observation["joint_action"]
        joint_action["arm"] = joint_action.pop("left_arm")
        joint_action["gripper"] = joint_action.pop("left_gripper")
        joint_action.pop("right_arm", None)
        joint_action.pop("right_gripper", None)
        joint_action["vector"] = np.concatenate(
            (np.asarray(joint_action["arm"], dtype=float), [joint_action["gripper"]])
        )
        endpose = observation.get("endpose", {})
        endpose["endpose"] = endpose.pop("left_endpose", None)
        endpose["gripper"] = endpose.pop("left_gripper", None)
        endpose.pop("right_endpose", None)
        endpose.pop("right_gripper", None)
        observation["observation"]["robot_state"] = self._actual_robot_state()
        step_index = int(self.simulation_step_count)
        observation["timing"] = {
            "simulation_step_index": np.int64(step_index),
            "simulation_time_s": np.float64(step_index * self.scene.get_timestep()),
        }
        return observation

    def gen_sparse_reward_data(self, chunk_actions, action_type="qpos"):
        """Execute a 7-D policy chunk through RoboTwin's existing TOPP path.

        A one-item RoboTwin embodiment aliases its legacy left/right handles to
        the same articulation.  Duplicating the command internally therefore
        drives one physical simulated robot twice to the same target; the model,
        dataset and public environment remain strictly seven dimensional.
        """
        actions = np.asarray(chunk_actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"single-Panthera actions must have shape [N, 7], got {actions.shape}")
        return super().gen_sparse_reward_data(
            np.concatenate((actions, actions), axis=1), action_type=action_type
        )

    def _move_arm(self, action, stage: str) -> bool:
        """Execute one single-arm action group and retain its milestone frame."""
        success = bool(self.move(action))
        self.capture_oracle_stage(stage)
        return success and self.plan_success

    def _advance_physics(self, simulation_steps: int):
        """Advance without sleep and preserve the configured continuous sample cadence."""
        for _ in range(simulation_steps):
            self._step_scene()
            if (
                self.save_data
                and self.save_freq is not None
                and self.save_freq > 0
                and self.simulation_step_count % self.save_freq == 0
            ):
                self._take_picture()

    def _top_down_grasp(self, arm: ArmTag, contact_point_id: int):
        """Build one fixed pregrasp/descent/close sequence for a known station."""
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
            Action(
                arm,
                "move",
                target_pose=grasp,
                constraint_pose=[1, 1, 1, 0, 0, 0],
            ),
            Action(arm, "close", target_gripper_pos=CYLINDER_GRASP_OPENING),
        ]

    def _carry_object_center_to(
        self,
        target_position: np.ndarray,
        stage: str,
        tolerance_m: float = 0.012,
        target_quaternion_wxyz: list[float] | None = None,
        max_segment_m: float | None = None,
    ) -> bool:
        """Move the wrist by measured object displacement at an optional attitude."""
        arm = ArmTag("left")
        target = np.asarray(target_position, dtype=float)
        initial_displacement = target - np.asarray(
            self.cylinder.get_pose().p, dtype=float
        )
        distance = float(np.linalg.norm(initial_displacement))
        segment_count = 1
        if max_segment_m is not None:
            if max_segment_m <= 0.0:
                raise ValueError("max_segment_m must be positive")
            segment_count = max(1, int(math.ceil(distance / max_segment_m)))
        for segment_index in range(segment_count):
            remaining = target - np.asarray(self.cylinder.get_pose().p, dtype=float)
            displacement = remaining / float(segment_count - segment_index)
            moved = self.move(
                self.move_by_displacement(
                    arm,
                    *displacement,
                    quat=target_quaternion_wxyz,
                )
            )
            if not moved or not self.plan_success:
                self.capture_oracle_stage(f"{stage}_planning_failed")
                return False
        self.capture_oracle_stage(stage)
        final_error = float(
            np.linalg.norm(target - np.asarray(self.cylinder.get_pose().p, dtype=float))
        )
        return final_error <= tolerance_m

    def play_once(self):
        """Run one-arm grasp, lift, align, release, settle and retract."""
        arm = ArmTag("left")

        if not self._move_arm(
            self.close_gripper(arm, pos=PREGRASP_OPENING), "preclose"
        ):
            return self.info
        if not self._move_arm(self._top_down_grasp(arm, 0), "grasped"):
            return self.info
        # Let the simulated contacts settle while the position controllers hold
        # the grasp.  This advances physics deterministically; it is not sleep.
        self._advance_physics(25)
        self.capture_oracle_stage("grasp_settled")
        if len(self.get_gripper_actor_contact_position("panthera_cylinder")) < 2:
            self.capture_oracle_stage("grasp_contact_rejected")
            return self.info

        lift_height = float(self.groove_target_pose.p[2] + 0.05)
        if not self._carry_object_center_to(
            np.array([self.cylinder.get_pose().p[0], self.cylinder.get_pose().p[1], lift_height]),
            "lifted",
        ):
            return self.info
        carry_height = float(self.groove_target_pose.p[2] + 0.05)
        above_groove_position = np.array(
            [self.groove_target_pose.p[0], self.groove_target_pose.p[1], carry_height]
        )
        if not self._carry_object_center_to(
            above_groove_position,
            "aligned_above_groove",
        ):
            return self.info

        # Stop above the rails, open the one gripper, then let gravity perform
        # the final insertion without any scripted object attachment.
        release_position = np.array(self.groove_target_pose.p) + np.array([0.0, 0.0, 0.03])
        if not self._carry_object_center_to(
            release_position,
            "release_height",
        ):
            return self.info
        if not self._move_arm(self.open_gripper(arm), "opened_for_release"):
            return self.info
        self._advance_physics(1)
        self.capture_oracle_stage("released")

        if not self._move_arm(
            self.move_by_displacement(arm, z=0.08), "gripper_withdrawn"
        ):
            return self.info

        # This advances exactly one simulated second at the configured 250 Hz.
        # It verifies post-release stability without a wall-clock sleep.
        self._advance_physics(SETTLE_SIMULATION_STEPS)
        self.capture_oracle_stage("released_and_settled")
        self._move_arm(self.back_to_origin(arm), "retracted")

        self.info["info"] = {"{A}": "cylinder", "{B}": "groove"}
        self.info["panthera_episode"] = {
            "schema_version": TASK_SCHEMA_VERSION,
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
            "groove_inner_half_width_m": GROOVE_INNER_HALF_WIDTH_M,
            "groove_rail_half_length_m": GROOVE_RAIL_HALF_LENGTH_M,
            "sample_period_physics_steps": int(self.save_freq),
            "attach_on_grasp": False,
        }
        self._terminal_success = self._metrics_pass(self.success_metrics())
        return self.info

    def success_metrics(self) -> dict[str, float | bool]:
        """Return explicit position, axis, velocity and gripper checks for acceptance."""
        pose = self.cylinder.get_pose()
        position_error = np.asarray(pose.p) - np.asarray(self.groove_target_pose.p)
        cylinder_axis = t3d.quaternions.quat2mat(pose.q)[:, 0]
        axis_dot = float(np.clip(abs(np.dot(cylinder_axis, [1.0, 0.0, 0.0])), 0.0, 1.0))
        axis_error_deg = math.degrees(math.acos(axis_dot))

        rigid_body = next(
            component
            for component in self.cylinder.actor.get_components()
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
        )
        linear_speed = float(np.linalg.norm(rigid_body.get_linear_velocity()))
        angular_speed = float(np.linalg.norm(rigid_body.get_angular_velocity()))
        return {
            "x_error_m": float(abs(position_error[0])),
            "lateral_error_m": float(abs(position_error[1])),
            "height_error_m": float(abs(position_error[2])),
            "axis_error_deg": float(axis_error_deg),
            "linear_speed_mps": linear_speed,
            "angular_speed_radps": angular_speed,
            "gripper_open": bool(self.is_left_gripper_open()),
        }

    @staticmethod
    def _metrics_pass(metrics: dict[str, float | bool]) -> bool:
        """Apply the terminal placement thresholds to one metrics snapshot."""
        return bool(
            metrics["x_error_m"] <= 0.020
            and metrics["lateral_error_m"] <= 0.012
            and metrics["height_error_m"] <= 0.012
            and metrics["axis_error_deg"] <= 6.0
            and metrics["linear_speed_mps"] <= 0.025
            and metrics["angular_speed_radps"] <= 0.35
            and metrics["gripper_open"]
        )

    def check_success(self):
        """Return the live or pre-close terminal placement result."""
        if self._terminal_success is not None:
            return self._terminal_success
        if self.robot is None:
            return False
        return self._metrics_pass(self.success_metrics())

    def get_info(self):
        """Provide the placeholders used by RoboTwin language generation."""
        return {"{A}": "cylinder", "{B}": "groove"}
