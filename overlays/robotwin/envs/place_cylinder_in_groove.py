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
from .panthera_release_reward import evaluate_transition
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
ROBOTWIN_SOURCE_COMMIT = "6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755"


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
        self.rl_reward_variant = str(
            kwargs.get("rl_reward_variant", "legacy")
        ).lower()
        if self.rl_reward_variant not in {"legacy", "r1", "r2"}:
            raise ValueError(
                "rl_reward_variant must be one of legacy, r1 or r2"
            )
        self._rl_previous_action = None
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
        """Read position, motion and transmitted load from one articulation.

        ``vector`` deliberately remains the historical seven-dimensional
        position-only observation so existing checkpoints keep working.
        ``dynamics_vector`` is the new 28-dimensional contract, ordered as
        qpos, qvel, qacc and effort; every block contains joint1..joint6 and
        the normalized gripper degree of freedom.

        PhysX reports the spatial force transmitted from a joint's parent to
        its child in the incoming joint frame.  Panthera's revolute axis is the
        joint-frame x axis, so arm effort is torque-x.  The finger joint is
        prismatic on x, so its effort is force-x.  This is intentionally not
        ``get_qf``: RoboTwin writes gravity/Coriolis compensation into qf, which
        is an applied command rather than measured joint load.
        """
        entity = self.robot.left_entity
        active_joints = entity.get_active_joints()
        active_qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        active_qvel = np.asarray(entity.get_qvel(), dtype=np.float64)
        active_qacc = np.asarray(entity.get_qacc(), dtype=np.float64)
        if not (
            active_qpos.shape == active_qvel.shape == active_qacc.shape
            == (len(active_joints),)
        ):
            raise RuntimeError("articulation state width does not match active joints")

        def by_name(values: np.ndarray) -> dict[str, float]:
            return {
                joint.get_name(): float(values[index])
                for index, joint in enumerate(active_joints)
            }

        qpos_by_name = by_name(active_qpos)
        qvel_by_name = by_name(active_qvel)
        qacc_by_name = by_name(active_qacc)
        arm_qpos = np.asarray(
            [qpos_by_name[joint.get_name()] for joint in self.robot.left_arm_joints],
            dtype=np.float64,
        )
        arm_qvel = np.asarray(
            [qvel_by_name[joint.get_name()] for joint in self.robot.left_arm_joints],
            dtype=np.float64,
        )
        arm_qacc = np.asarray(
            [qacc_by_name[joint.get_name()] for joint in self.robot.left_arm_joints],
            dtype=np.float64,
        )
        gripper_joint = self.robot.left_gripper[0][0]
        finger_qpos = qpos_by_name[gripper_joint.get_name()]
        finger_qvel = qvel_by_name[gripper_joint.get_name()]
        finger_qacc = qacc_by_name[gripper_joint.get_name()]
        scale = np.asarray(self.robot.left_gripper_scale, dtype=float)
        span = float(scale[1] - scale[0])
        if not np.isfinite(span) or abs(span) <= 1.0e-12:
            raise RuntimeError("gripper scale must have a finite nonzero span")
        gripper_opening = float(
            np.clip((finger_qpos - scale[0]) / span, 0.0, 1.0)
        )
        gripper_velocity = float(finger_qvel / span)
        gripper_acceleration = float(finger_qacc / span)

        incoming = np.asarray(entity.get_link_incoming_joint_forces(), dtype=np.float64)
        if incoming.ndim != 2 or incoming.shape[1] != 6:
            raise RuntimeError(f"unexpected incoming-joint-force shape {incoming.shape}")
        arm_effort = np.asarray(
            [incoming[joint.get_child_link().get_index(), 3]
             for joint in self.robot.left_arm_joints],
            dtype=np.float64,
        )
        gripper_effort = float(
            incoming[gripper_joint.get_child_link().get_index(), 0]
        )
        position = np.concatenate((arm_qpos, [gripper_opening]))
        velocity = np.concatenate((arm_qvel, [gripper_velocity]))
        acceleration = np.concatenate((arm_qacc, [gripper_acceleration]))
        effort = np.concatenate((arm_effort, [gripper_effort]))
        dynamics = np.concatenate((position, velocity, acceleration, effort))
        if not np.all(np.isfinite(dynamics)):
            raise RuntimeError("robot dynamics state contains NaN or infinity")
        return {
            "arm_qpos": arm_qpos,
            "gripper_qpos": gripper_opening,
            "arm_qvel": arm_qvel,
            "gripper_qvel": gripper_velocity,
            "arm_qacc": arm_qacc,
            "gripper_qacc": gripper_acceleration,
            "arm_effort": arm_effort,
            "gripper_effort": gripper_effort,
            "vector": position,
            "dynamics_vector": dynamics,
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

    def _reset_policy_episode_metrics(self) -> None:
        """Reset privileged diagnostics for one policy-controlled episode.

        These values are exported only through the environment ``info``
        channel.  They are deliberately absent from the RGB/proprioception
        observation and are never read by the reward evaluator.
        """
        self._rl_metric_grasped_once = False
        self._rl_metric_target_grasped_once = False
        self._rl_metric_valid_release_once = False
        self._rl_metric_invalid_release_once = False
        self._rl_metric_hard_failure_once = False
        self._rl_metric_first_target_grasped_action = None
        self._rl_metric_first_valid_release_action = None
        self._rl_metric_arm_delta_sum = 0.0
        self._rl_metric_arm_delta_count = 0
        self._rl_metric_arm_second_difference_sum = 0.0
        self._rl_metric_arm_second_difference_max = 0.0
        self._rl_metric_arm_second_difference_count = 0
        self._rl_metric_boundary_second_difference_sum = 0.0
        self._rl_metric_boundary_second_difference_max = 0.0
        self._rl_metric_boundary_second_difference_count = 0
        self._rl_metric_gripper_delta_sum = 0.0
        self._rl_metric_gripper_delta_count = 0
        self._rl_metric_previous_arm_delta = None
        self._rl_metric_last_gripper_target = None

    @staticmethod
    def _policy_effect_is_release_aligned(effect) -> bool:
        """Return whether the object geometry is inside the release envelope."""
        return bool(
            effect.target_xy_error_m <= 0.030
            and effect.target_height_error_m <= 0.080
            and effect.target_axis_error_deg <= 10.0
        )

    def _record_policy_effect(self, previous, current) -> None:
        """Accumulate physical transition outcomes without shaping policy reward."""
        action_index = int(self.take_action_cnt)
        aligned_while_grasped = bool(
            current.grasped and self._policy_effect_is_release_aligned(current)
        )
        self._rl_metric_grasped_once |= bool(current.grasped)
        self._rl_metric_target_grasped_once |= aligned_while_grasped
        self._rl_metric_valid_release_once |= bool(current.released_in_valid_volume)
        self._rl_metric_hard_failure_once |= bool(current.hard_failure)
        if (
            aligned_while_grasped
            and self._rl_metric_first_target_grasped_action is None
        ):
            self._rl_metric_first_target_grasped_action = action_index
        if (
            current.released_in_valid_volume
            and self._rl_metric_first_valid_release_action is None
        ):
            self._rl_metric_first_valid_release_action = action_index
        if (
            previous is not None
            and previous.grasped
            and not current.grasped
            and not current.released_in_valid_volume
        ):
            self._rl_metric_invalid_release_once = True

    def _record_policy_action_motion(
        self,
        *,
        target_arm: np.ndarray,
        previous_arm: np.ndarray,
        target_gripper: float,
        previous_gripper: float,
        is_chunk_start: bool,
    ) -> None:
        """Measure target changes and second differences at the 50 Hz contract."""
        arm_delta = np.asarray(target_arm, dtype=float) - np.asarray(
            previous_arm, dtype=float
        )
        arm_delta_l1 = float(np.mean(np.abs(arm_delta)))
        self._rl_metric_arm_delta_sum += arm_delta_l1
        self._rl_metric_arm_delta_count += 1
        self._rl_metric_gripper_delta_sum += abs(
            float(target_gripper) - float(previous_gripper)
        )
        self._rl_metric_gripper_delta_count += 1

        previous_delta = self._rl_metric_previous_arm_delta
        if previous_delta is not None:
            second_difference = float(np.mean(np.abs(arm_delta - previous_delta)))
            self._rl_metric_arm_second_difference_sum += second_difference
            self._rl_metric_arm_second_difference_max = max(
                self._rl_metric_arm_second_difference_max,
                second_difference,
            )
            self._rl_metric_arm_second_difference_count += 1
            if is_chunk_start:
                self._rl_metric_boundary_second_difference_sum += second_difference
                self._rl_metric_boundary_second_difference_max = max(
                    self._rl_metric_boundary_second_difference_max,
                    second_difference,
                )
                self._rl_metric_boundary_second_difference_count += 1
        self._rl_metric_previous_arm_delta = arm_delta.copy()
        self._rl_metric_last_gripper_target = float(target_gripper)

    @staticmethod
    def _safe_mean(total: float, count: int) -> float:
        return float(total / count) if count else 0.0

    def _policy_episode_metrics(self, final_effect) -> dict[str, float]:
        """Return cumulative, scalar episode metrics suitable for RLinf logging."""
        first_target = self._rl_metric_first_target_grasped_action
        first_release = self._rl_metric_first_valid_release_action
        release_delay = 0
        release_delay_censored = False
        if first_target is not None:
            if first_release is None:
                release_delay = max(0, int(self.take_action_cnt) - first_target)
                release_delay_censored = True
            else:
                release_delay = max(0, first_release - first_target)
        aligned_grasped_at_end = bool(
            final_effect.grasped
            and self._policy_effect_is_release_aligned(final_effect)
        )
        return {
            "panthera_grasped_once": float(self._rl_metric_grasped_once),
            "panthera_target_grasped_once": float(
                self._rl_metric_target_grasped_once
            ),
            "panthera_valid_release_once": float(
                self._rl_metric_valid_release_once
            ),
            "panthera_invalid_release_once": float(
                self._rl_metric_invalid_release_once
            ),
            "panthera_hard_failure_once": float(
                self._rl_metric_hard_failure_once
            ),
            "panthera_target_still_grasped_at_end": float(
                aligned_grasped_at_end
            ),
            "panthera_valid_release_at_end": float(
                final_effect.released_in_valid_volume
            ),
            "panthera_release_delay_actions": float(release_delay),
            "panthera_release_delay_censored": float(release_delay_censored),
            "panthera_arm_delta_l1_mean": self._safe_mean(
                self._rl_metric_arm_delta_sum,
                self._rl_metric_arm_delta_count,
            ),
            "panthera_arm_second_difference_l1_mean": self._safe_mean(
                self._rl_metric_arm_second_difference_sum,
                self._rl_metric_arm_second_difference_count,
            ),
            "panthera_arm_second_difference_l1_max": float(
                self._rl_metric_arm_second_difference_max
            ),
            "panthera_chunk_boundary_second_difference_l1_mean": self._safe_mean(
                self._rl_metric_boundary_second_difference_sum,
                self._rl_metric_boundary_second_difference_count,
            ),
            "panthera_chunk_boundary_second_difference_l1_max": float(
                self._rl_metric_boundary_second_difference_max
            ),
            "panthera_gripper_delta_l1_mean": self._safe_mean(
                self._rl_metric_gripper_delta_sum,
                self._rl_metric_gripper_delta_count,
            ),
            "panthera_target_xy_error_m": float(final_effect.target_xy_error_m),
            "panthera_target_height_error_m": float(
                final_effect.target_height_error_m
            ),
            "panthera_target_axis_error_deg": float(
                final_effect.target_axis_error_deg
            ),
        }

    def gen_sparse_reward_data(self, chunk_actions, action_type="qpos"):
        """Execute one 50 Hz, 7-D policy chunk and return sparse task reward.

        RoboTwin ``6dde571`` removed the legacy base-class
        ``gen_sparse_reward_data`` API.  Panthera actions are already dense
        absolute joint targets, so replay them with the same timing contract
        as :class:`packages.panthera_sim.executor.DenseExecutor`: five 250 Hz
        physics ticks per policy target and finite-difference velocity
        feed-forward.  This keeps RL rollout dynamics aligned with the expert
        replay gate instead of routing each sample through a fresh planner.
        """
        if action_type != "qpos":
            raise ValueError("single-Panthera sparse rollout only supports qpos actions")
        actions = np.asarray(chunk_actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"single-Panthera actions must have shape [N, 7], got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("single-Panthera actions must be finite")

        reward_variant = getattr(self, "rl_reward_variant", "legacy")
        use_effect_reward = (
            reward_variant in {"r1", "r2"}
            and hasattr(self, "policy_release_effect_state")
        )
        previous_effect = (
            self.policy_release_effect_state(update_settle=False)
            if use_effect_reward
            else None
        )
        if use_effect_reward:
            self._record_policy_effect(None, previous_effect)
        info = {"success": bool(getattr(self, "eval_success", False))}
        reward = np.array([float(info["success"])], dtype=np.float32)
        termination = np.array([int(info["success"])], dtype=np.int32)
        truncation = np.array([0], dtype=np.int32)
        if info["success"]:
            info["executed_steps"] = 1
            if use_effect_reward:
                info["step_rewards"] = [0.0]
                info.update(self._policy_episode_metrics(previous_effect))
            return reward, termination, truncation, info

        remaining = max(0, int(self.step_lim) - int(self.take_action_cnt))
        if remaining == 0:
            truncation[0] = 1
            info["executed_steps"] = 1
            if use_effect_reward:
                info["step_rewards"] = [0.0]
                info.update(self._policy_episode_metrics(previous_effect))
            return reward, termination, truncation, info

        actions = actions[:remaining]
        measured = self._actual_robot_state()
        previous_arm = np.asarray(measured["arm_qpos"], dtype=float)
        previous_gripper = float(measured["gripper_qpos"])
        physics_steps_per_action = 5
        policy_period_s = physics_steps_per_action / 250.0
        executed_steps = 0
        step_rewards: list[float] = []
        reward_components: list[dict[str, float]] = []

        for action_offset, target in enumerate(actions):
            target_arm = np.asarray(target[:6], dtype=float)
            target_gripper = float(np.clip(target[6], 0.0, 1.0))
            target_velocity = (target_arm - previous_arm) / policy_period_s
            if use_effect_reward:
                self._record_policy_action_motion(
                    target_arm=target_arm,
                    previous_arm=previous_arm,
                    target_gripper=target_gripper,
                    previous_gripper=previous_gripper,
                    is_chunk_start=action_offset == 0,
                )

            # Expert replay applies the task's release-contact calibration on
            # the first closed-to-open transition.  Keep the same transition
            # in policy rollout so success is not decided by executor drift.
            if (
                previous_gripper < 0.5 <= target_gripper
                and not getattr(self, "_policy_release_solver_configured", False)
                and hasattr(self, "_configure_release_contact_solver")
            ):
                self._configure_release_contact_solver()
                self._policy_release_solver_configured = True

            for _ in range(physics_steps_per_action):
                self.robot.set_arm_joints(target_arm, target_velocity, "left")
                self.robot.set_gripper(target_gripper, "left")
                self._step_scene()

            self.take_action_cnt += 1
            executed_steps += 1
            previous_arm = target_arm
            previous_gripper = target_gripper

            if use_effect_reward:
                current_effect = self.policy_release_effect_state(update_settle=True)
                self._record_policy_effect(previous_effect, current_effect)
                transition = evaluate_transition(
                    previous_effect,
                    current_effect,
                    variant=reward_variant,
                    previous_action=self._rl_previous_action,
                    current_action=target,
                )
                step_rewards.append(transition.total)
                reward_components.append(
                    {
                        "task": transition.task,
                        "shaping": transition.shaping,
                        "time": transition.time,
                        "arm_action_change": transition.arm_action_change,
                        "previous_potential": transition.previous_potential,
                        "next_potential": transition.next_potential,
                    }
                )
                self._rl_previous_action = target.copy()
                previous_effect = current_effect
                if current_effect.success:
                    self.eval_success = True
                    info["success"] = True
                    termination[0] = 1
                    info["termination_reason"] = "stable_insertion"
                    break
                if current_effect.hard_failure:
                    termination[0] = 1
                    info["hard_failure"] = True
                    info["termination_reason"] = "unrecoverable_object_state"
                    break
            elif self.check_success():
                self.eval_success = True
                info["success"] = True
                reward[0] = 1.0
                termination[0] = 1
                break

        if self.take_action_cnt >= self.step_lim and not info["success"]:
            truncation[0] = 1
        info["executed_steps"] = max(1, executed_steps)
        if use_effect_reward:
            if not step_rewards:
                step_rewards = [0.0]
            reward[0] = float(sum(step_rewards))
            info["step_rewards"] = step_rewards
            info["reward_components"] = reward_components
            info.update(self._policy_episode_metrics(previous_effect))
        return reward, termination, truncation, info

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
