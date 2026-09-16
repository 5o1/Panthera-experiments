"""Panthera v2 task with broad placement and upright/lying cylinder poses."""

from __future__ import annotations

from copy import deepcopy
import math
import os

import numpy as np
from scipy.interpolate import CubicSpline
import transforms3d as t3d

from .panthera_v2_sampling import sample_scene
from .place_cylinder_in_groove import TABLE_HEIGHT_M
from .place_vertical_cylinder_in_groove import (
    CYLINDER_HALF_HEIGHT_M,
    CYLINDER_RADIUS_M,
    PREGRASP_OPENING,
    SETTLE_SIMULATION_STEPS,
    TOP_DOWN_GRASP_QUATERNION_WXYZ,
    place_vertical_cylinder_in_groove,
)
from .utils import Action, ArmTag


V2_SCHEMA_VERSION = 10
V2_SCENE_PROFILE = "panthera_phone_symmetric_single_grasp_direct_release_cylinder_socket_v2"
SAFE_CARRY_Z_M = 0.905
REORIENTATION_STEPS = 6
V2_GRASP_OPENING = 0.30
REORIENTATION_STATION_XY_M = np.array([-0.18, -0.07], dtype=float)
REGRASP_STATION_XY_M = np.array([-0.24, -0.07], dtype=float)
HIGH_LIFT_STATION_XY_M = np.array([-0.24, -0.16], dtype=float)
MIN_REORIENTATION_RELEASE_Z_M = 0.81
REORIENTATION_RELEASE_Z_M = 0.86
REGRASP_TRANSFER_Z_M = 0.85
CONTINUOUS_SPEED_SCALE = 0.60
CONTINUOUS_CONTROL_FREQUENCY_HZ = 250.0
CONTINUOUS_MAX_JOINT_VELOCITY_RADPS = 0.60
CONTINUOUS_MAX_JOINT_ACCELERATION_RADPS2 = 2.0
CONTINUOUS_RETIME_MAX_ITERATIONS = 8
MOTION_SETTLE_MAX_STEPS = 250
MOTION_SETTLE_WINDOW_STEPS = 50
MOTION_SETTLE_MAX_JOINT_SPAN_RAD = 0.005
JOINT_BLEND_INPUT_RESOLUTION_RAD = 0.010
JOINT_BLEND_ITERATIONS = 4
UPRIGHT_SPLINE_CONTROL_RESOLUTION_RAD = 0.120
LYING_SPLINE_CONTROL_RESOLUTION_RAD = 0.200
JOINT_SPLINE_DENSE_SAMPLES_PER_CONTROL = 16
DIRECT_RELEASE_BOTTOM_CLEARANCE_M = -0.014
DIRECT_RELEASE_MIN_CLEARANCE_M = -0.035
DIRECT_RELEASE_MAX_CLEARANCE_M = 0.020
LYING_END_GRASP_OFFSET_M = 0.044
LYING_END_GRASP_MAX_OFFSET_M = CYLINDER_HALF_HEIGHT_M - 0.015
DIRECT_RELEASE_SETTLE_MAX_STEPS = 10 * SETTLE_SIMULATION_STEPS
CYLINDER_LINEAR_DAMPING = 1.0
CYLINDER_ANGULAR_DAMPING = 2.0
CYLINDER_SOLVER_POSITION_ITERATIONS = 20
CYLINDER_SOLVER_VELOCITY_ITERATIONS = 4
CYLINDER_MAX_DEPENETRATION_VELOCITY_MPS = 0.2
DIRECT_RELEASE_LOWEST_VALIDATED_CLEARANCE_M = -0.015
DIRECT_RELEASE_CLEARANCE_MARGIN_M = 0.001


def _mirrored_station(base_x_m: float, object_x_m: float, station: np.ndarray) -> np.ndarray:
    """Place a manipulation station on the object's side of the robot."""
    side = -1.0 if object_x_m < base_x_m else 1.0
    return np.array([base_x_m + side * abs(station[0]), station[1]], dtype=float)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Normalize one finite wxyz quaternion."""
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-10:
        raise ValueError("quaternion must be nonzero")
    return value / norm


def _slerp_wxyz(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate along the shortest unit-quaternion arc."""
    first = _normalize_quaternion(start)
    second = _normalize_quaternion(end)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(first + fraction * (second - first))
    angle = math.acos(dot)
    scale = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / scale * first
        + math.sin(fraction * angle) / scale * second
    )


def _lying_grasp_quaternion(angle_rad: float) -> list[float]:
    """Approach from above and close perpendicular to a lying cylinder axis."""
    axis = np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0])
    approach = np.array([0.0, 0.0, -1.0])
    closing = np.array([-math.sin(angle_rad), math.cos(angle_rad), 0.0])
    rotation = np.column_stack((approach, closing, axis))
    return _normalize_quaternion(t3d.quaternions.mat2quat(rotation)).tolist()


def _upright_side_grasp_quaternion(
    radial_xy: np.ndarray,
    cylinder_axis_sign: float = 1.0,
) -> list[float]:
    """Keep the held cylinder vertical with its grasped end pointing upward."""
    radial = np.asarray(radial_xy, dtype=float)
    radial /= np.linalg.norm(radial)
    tool_x = np.array([radial[0], radial[1], 0.0])
    tool_z = np.array([0.0, 0.0, math.copysign(1.0, cylinder_axis_sign)])
    tool_y = np.cross(tool_z, tool_x)
    rotation = np.column_stack((tool_x, tool_y, tool_z))
    return _normalize_quaternion(t3d.quaternions.mat2quat(rotation)).tolist()


class place_randomized_cylinder_in_socket(place_vertical_cylinder_in_groove):
    """Insert a broadly placed upright or lying cylinder using one Panthera."""

    task_schema_version = V2_SCHEMA_VERSION
    scene_profile = V2_SCENE_PROFILE

    def setup_demo(self, **kwargs):
        """Configure explicit cylinder damping before an oracle episode."""
        # RoboTwin reuses this task object while collecting several episodes.
        # These records belong to one episode and must not leak into the next
        # scene_info entry.
        self.continuous_motion_audit = []
        self.motion_settle_audit = []
        self.joint_smoothing_audit = []
        self.joint_retime_audit = []
        self.grasp_route_selection_audit = []
        linear_damping = float(
            os.environ.get(
                "PANTHERA_V2_CYLINDER_LINEAR_DAMPING",
                str(CYLINDER_LINEAR_DAMPING),
            )
        )
        angular_damping = float(
            os.environ.get(
                "PANTHERA_V2_CYLINDER_ANGULAR_DAMPING",
                str(CYLINDER_ANGULAR_DAMPING),
            )
        )
        if not 0.0 <= linear_damping <= 10.0:
            raise ValueError("cylinder linear damping must be in [0, 10]")
        if not 0.0 <= angular_damping <= 10.0:
            raise ValueError("cylinder angular damping must be in [0, 10]")
        super().setup_demo(**kwargs)
        rigid_body = next(
            component
            for component in self.cylinder.actor.get_components()
            if hasattr(component, "set_linear_damping")
        )
        self.release_linear_damping = linear_damping
        self.release_angular_damping = angular_damping
        position_iterations = int(
            os.environ.get(
                "PANTHERA_V2_SOLVER_POSITION_ITERATIONS",
                str(CYLINDER_SOLVER_POSITION_ITERATIONS),
            )
        )
        velocity_iterations = int(
            os.environ.get(
                "PANTHERA_V2_SOLVER_VELOCITY_ITERATIONS",
                str(CYLINDER_SOLVER_VELOCITY_ITERATIONS),
            )
        )
        max_depenetration_velocity = float(
            os.environ.get(
                "PANTHERA_V2_MAX_DEPENETRATION_VELOCITY_MPS",
                str(CYLINDER_MAX_DEPENETRATION_VELOCITY_MPS),
            )
        )
        if not 1 <= position_iterations <= 255:
            raise ValueError("solver position iterations must be in [1, 255]")
        if not 1 <= velocity_iterations <= 255:
            raise ValueError("solver velocity iterations must be in [1, 255]")
        if not 0.0 < max_depenetration_velocity <= 10.0:
            raise ValueError("max depenetration velocity must be in (0, 10]")
        self.release_solver_position_iterations = position_iterations
        self.release_solver_velocity_iterations = velocity_iterations
        self.release_max_depenetration_velocity_mps = max_depenetration_velocity
        self.physics_parameters["cylinder_linear_damping"] = linear_damping
        self.physics_parameters["cylinder_angular_damping"] = angular_damping
        self.physics_parameters["cylinder_solver_position_iterations"] = (
            position_iterations
        )
        self.physics_parameters["cylinder_solver_velocity_iterations"] = (
            velocity_iterations
        )
        self.physics_parameters["cylinder_max_depenetration_velocity_mps"] = (
            max_depenetration_velocity
        )

    def _configure_release_contact_solver(self):
        """Apply tighter contact solving only after the held route is complete."""
        rigid_body = next(
            component
            for component in self.cylinder.actor.get_components()
            if hasattr(component, "set_solver_position_iterations")
        )
        rigid_body.set_linear_damping(self.release_linear_damping)
        rigid_body.set_angular_damping(self.release_angular_damping)
        rigid_body.set_solver_position_iterations(
            self.release_solver_position_iterations
        )
        rigid_body.set_solver_velocity_iterations(
            self.release_solver_velocity_iterations
        )
        rigid_body.set_max_depenetration_velocity(
            self.release_max_depenetration_velocity_mps
        )

    def _smooth_and_validate_joint_path(
        self,
        geometric_path: np.ndarray,
        stage: str,
    ) -> np.ndarray | None:
        """Round joint-path corners with convex local blends and check collisions."""
        path = np.asarray(geometric_path, dtype=float)
        segment_length = np.linalg.norm(np.diff(path, axis=0), axis=1)
        keep = np.concatenate((np.array([True]), segment_length > 1.0e-6))
        path = path[keep]
        if len(path) < 2:
            return None
        arc_length = np.concatenate(
            (np.array([0.0]), np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )
        total_length = float(arc_length[-1])
        if total_length <= 1.0e-8:
            return None
        sample_count = max(
            2,
            int(math.ceil(total_length / JOINT_BLEND_INPUT_RESOLUTION_RAD)) + 1,
        )
        sample_arc = np.linspace(0.0, total_length, sample_count)
        smoothed = np.column_stack(
            [
                np.interp(sample_arc, arc_length, path[:, joint])
                for joint in range(path.shape[1])
            ]
        )
        for _ in range(JOINT_BLEND_ITERATIONS):
            first = 0.75 * smoothed[:-1] + 0.25 * smoothed[1:]
            second = 0.25 * smoothed[:-1] + 0.75 * smoothed[1:]
            blended = np.empty((2 * (len(smoothed) - 1), smoothed.shape[1]))
            blended[0::2] = first
            blended[1::2] = second
            smoothed = np.vstack((smoothed[0], blended, smoothed[-1]))
        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
        if not np.all(np.isfinite(smoothed)):
            return None

        planner = getattr(self.robot.left_planner, "planner", None)
        qpos_template = np.asarray(self.robot.left_entity.get_qpos(), dtype=float)
        collision_samples = np.unique(
            np.concatenate((np.arange(0, len(smoothed), 2), [len(smoothed) - 1]))
        )
        if planner is not None and hasattr(planner, "check_for_self_collision"):
            for index in collision_samples:
                qpos = qpos_template.copy()
                qpos[: smoothed.shape[1]] = smoothed[index]
                if planner.check_for_self_collision(qpos) or planner.check_for_env_collision(qpos):
                    self.capture_oracle_stage(f"{stage}_smoothed_collision")
                    return None

        tangent = np.diff(smoothed, axis=0)
        tangent_norm = np.linalg.norm(tangent, axis=1)
        valid = tangent_norm > 1.0e-8
        unit_tangent = tangent[valid] / tangent_norm[valid, None]
        if len(unit_tangent) > 1:
            cosine = np.clip(
                np.sum(unit_tangent[:-1] * unit_tangent[1:], axis=1), -1.0, 1.0
            )
            maximum_tangent_step_deg = float(
                np.degrees(np.max(np.arccos(cosine)))
            )
        else:
            maximum_tangent_step_deg = 0.0
        if not hasattr(self, "joint_smoothing_audit"):
            self.joint_smoothing_audit = []
        self.joint_smoothing_audit.append(
            {
                "stage": stage,
                "input_points": int(len(path)),
                "output_points": int(len(smoothed)),
                "maximum_tangent_step_deg": maximum_tangent_step_deg,
                "collision_samples": int(len(collision_samples)),
                "blend_iterations": JOINT_BLEND_ITERATIONS,
            }
        )
        return smoothed

    def _retime_joint_path_by_arc_length(
        self,
        geometric_path: np.ndarray,
        stage: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Apply one monotonic quintic time law to the complete joint path."""
        path = np.asarray(geometric_path, dtype=float)
        segment_length = np.linalg.norm(np.diff(path, axis=0), axis=1)
        keep = np.concatenate((np.array([True]), segment_length > 1.0e-8))
        path = path[keep]
        if len(path) < 2 or not np.all(np.isfinite(path)):
            return None
        arc = np.concatenate(
            (np.array([0.0]), np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        )
        total_arc = float(arc[-1])
        if total_arc <= 1.0e-8:
            return None

        control_resolution = (
            LYING_SPLINE_CONTROL_RESOLUTION_RAD
            if self.cylinder_posture == "lying"
            else UPRIGHT_SPLINE_CONTROL_RESOLUTION_RAD
        )
        control_count = max(
            4,
            int(math.ceil(total_arc / control_resolution)) + 1,
        )
        control_arc = np.linspace(0.0, total_arc, control_count)
        controls = np.column_stack(
            [
                np.interp(control_arc, arc, path[:, joint])
                for joint in range(path.shape[1])
            ]
        )
        control_parameter = control_arc / total_arc
        spline = CubicSpline(
            control_parameter,
            controls,
            axis=0,
            bc_type="natural",
        )
        dense_parameter = np.linspace(
            0.0,
            1.0,
            max(
                1000,
                JOINT_SPLINE_DENSE_SAMPLES_PER_CONTROL * control_count,
            ),
        )
        dense_path = np.asarray(spline(dense_parameter), dtype=float)
        dense_arc = np.concatenate(
            (
                np.array([0.0]),
                np.cumsum(np.linalg.norm(np.diff(dense_path, axis=0), axis=1)),
            )
        )
        spline_arc = float(dense_arc[-1])
        if spline_arc <= 1.0e-8 or not np.all(np.isfinite(dense_path)):
            return None

        # Quintic smoothstep has a maximum normalized speed of 1.875.
        duration = max(
            1.0,
            1.875 * spline_arc / CONTINUOUS_MAX_JOINT_VELOCITY_RADPS,
        )
        positions = velocities = accelerations = times = None
        peak_velocity = peak_acceleration = math.inf
        for iteration in range(1, CONTINUOUS_RETIME_MAX_ITERATIONS + 1):
            sample_count = max(
                3,
                int(math.ceil(duration * CONTINUOUS_CONTROL_FREQUENCY_HZ)) + 1,
            )
            times = np.linspace(0.0, duration, sample_count)
            phase = times / duration
            progress = (
                10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
            ) * spline_arc
            path_parameter = np.interp(
                progress, dense_arc, dense_parameter
            )
            positions = np.asarray(spline(path_parameter), dtype=float)
            velocities = np.gradient(
                positions, times, axis=0, edge_order=2
            )
            accelerations = np.gradient(
                velocities, times, axis=0, edge_order=2
            )
            peak_velocity = float(np.max(np.abs(velocities)))
            peak_acceleration = float(np.max(np.abs(accelerations)))
            scale = max(
                1.0,
                peak_velocity / CONTINUOUS_MAX_JOINT_VELOCITY_RADPS,
                math.sqrt(
                    peak_acceleration
                    / CONTINUOUS_MAX_JOINT_ACCELERATION_RADPS2
                ),
            )
            if scale <= 1.001:
                break
            duration *= scale * 1.02
        else:
            return None

        assert positions is not None
        assert velocities is not None
        assert accelerations is not None
        assert times is not None
        velocities[0] = 0.0
        velocities[-1] = 0.0

        command_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
        moving = np.flatnonzero(command_speed >= 0.02)
        if len(moving) < 2:
            return None
        interior = command_speed[moving[0] : moving[-1] + 1]
        low_speed = interior < 0.02
        longest_low_run_steps = 0
        current_run_steps = 0
        for is_low in low_speed:
            if is_low:
                current_run_steps += 1
                longest_low_run_steps = max(
                    longest_low_run_steps, current_run_steps
                )
            else:
                current_run_steps = 0
        longest_low_run_s = longest_low_run_steps / CONTINUOUS_CONTROL_FREQUENCY_HZ
        if longest_low_run_s >= 0.08:
            return None

        planner = getattr(self.robot.left_planner, "planner", None)
        qpos_template = np.asarray(self.robot.left_entity.get_qpos(), dtype=float)
        collision_indices = np.unique(
            np.concatenate((np.arange(0, len(positions), 2), [len(positions) - 1]))
        )
        if planner is not None and hasattr(planner, "check_for_self_collision"):
            for index in collision_indices:
                qpos = qpos_template.copy()
                qpos[: positions.shape[1]] = positions[index]
                if planner.check_for_self_collision(qpos) or planner.check_for_env_collision(qpos):
                    return None

        self.joint_retime_audit.append(
            {
                "stage": stage,
                "profile": "quintic_monotonic_arc_length",
                "spline_control_points": int(control_count),
                "spline_control_resolution_rad": control_resolution,
                "iterations": iteration,
                "trajectory_steps": int(len(positions)),
                "duration_s": float(times[-1]),
                "peak_joint_velocity_radps": peak_velocity,
                "peak_joint_acceleration_radps2": peak_acceleration,
                "longest_internal_near_zero_run_s": longest_low_run_s,
                "collision_samples": int(len(collision_indices)),
            }
        )
        return times, positions, velocities, accelerations

    def _execute_continuous_ee_waypoints(
        self,
        target_poses: list[list[float]],
        stage: str,
    ) -> bool:
        """Plan waypoint pieces, retime them together, and execute without stops."""
        if not target_poses:
            raise ValueError("continuous motion needs at least one target pose")
        if self.need_plan:
            path_chunks: list[np.ndarray] = []
            last_qpos = None
            for waypoint_index, target_pose in enumerate(target_poses):
                try:
                    result = self.robot.left_plan_path(
                        target_pose,
                        last_qpos=last_qpos,
                    )
                except Exception:
                    self.plan_success = False
                    self.capture_oracle_stage(
                        f"{stage}_waypoint_{waypoint_index}_planning_failed"
                    )
                    return False
                if result is None or result.get("status") != "Success":
                    self.plan_success = False
                    self.capture_oracle_stage(
                        f"{stage}_waypoint_{waypoint_index}_planning_failed"
                    )
                    return False
                positions = np.asarray(result["position"], dtype=float)
                if positions.ndim != 2 or len(positions) == 0:
                    self.plan_success = False
                    self.capture_oracle_stage(f"{stage}_empty_path")
                    return False
                if path_chunks and np.allclose(
                    path_chunks[-1][-1], positions[0], atol=1.0e-8
                ):
                    positions = positions[1:]
                if len(positions):
                    path_chunks.append(positions)
                last_qpos = np.asarray(result["position"][-1], dtype=float)
            geometric_path = self._smooth_and_validate_joint_path(
                np.vstack(path_chunks), stage
            )
            if geometric_path is None or len(geometric_path) < 2:
                self.plan_success = False
                self.capture_oracle_stage(f"{stage}_smoothing_failed")
                return False
            retimed = self._retime_joint_path_by_arc_length(
                geometric_path, stage
            )
            if retimed is None:
                self.plan_success = False
                self.capture_oracle_stage(f"{stage}_retiming_failed")
                return False
            scaled_times, positions, velocities, accelerations = retimed
            scaled_duration = float(scaled_times[-1])
            retime_audit = deepcopy(self.joint_retime_audit[-1])
            continuous_result = {
                "status": "Success",
                "time": scaled_times,
                "position": positions,
                "velocity": velocities,
                "acceleration": accelerations,
                "duration": scaled_duration,
                "cartesian_waypoint_count": len(target_poses),
                "motion_profile": "chaikin_arc_length_quintic_single_grasp_v3",
                "speed_scale": CONTINUOUS_SPEED_SCALE,
                "joint_retime_audit": retime_audit,
            }
            self.left_joint_path.append(deepcopy(continuous_result))
        else:
            continuous_result = deepcopy(self.left_joint_path[self.left_cnt])
            self.left_cnt += 1
            if continuous_result.get("status") != "Success":
                self.plan_success = False
                return False
            retime_audit = continuous_result.get("joint_retime_audit")
            if retime_audit is not None:
                self.joint_retime_audit.append(deepcopy(retime_audit))

        positions = np.asarray(continuous_result["position"], dtype=float)
        velocities = np.asarray(continuous_result["velocity"], dtype=float)
        if positions.ndim != 2 or velocities.shape != positions.shape:
            self.plan_success = False
            return False
        speed = np.linalg.norm(velocities, axis=1)
        interior_speed = speed[1:-1]
        audit = {
            "stage": stage,
            "cartesian_waypoint_count": int(
                continuous_result.get("cartesian_waypoint_count", len(target_poses))
            ),
            "trajectory_steps": int(len(positions)),
            "duration_s": float(continuous_result.get("duration", 0.0)),
            "speed_scale": float(
                continuous_result.get("speed_scale", CONTINUOUS_SPEED_SCALE)
            ),
            "interior_near_zero_speed_fraction": (
                float(np.mean(interior_speed < 1.0e-4))
                if len(interior_speed)
                else 0.0
            ),
        }
        if not hasattr(self, "continuous_motion_audit"):
            self.continuous_motion_audit = []
        audit["start_simulation_step"] = int(self.simulation_step_count)
        self.continuous_motion_audit.append(audit)
        self.take_dense_action(
            {
                "left_arm": continuous_result,
                "left_gripper": None,
                "right_arm": None,
                "right_gripper": None,
            }
        )
        audit["end_simulation_step"] = int(self.simulation_step_count)
        self.capture_oracle_stage(stage)
        return self.plan_success

    def _has_gripper_socket_contact(self) -> bool:
        """Return whether any gripper link touches the socket floor or walls."""
        socket_names = {
            "panthera_yellow_socket_floor",
            "panthera_yellow_socket_near_wall",
            "panthera_yellow_socket_far_wall",
            "panthera_yellow_socket_left_wall",
            "panthera_yellow_socket_right_wall",
        }
        gripper_names = set(self.robot.gripper_name)
        return any(
            {body.entity.name for body in contact.bodies}.intersection(socket_names)
            and {body.entity.name for body in contact.bodies}.intersection(gripper_names)
            for contact in self.scene.get_contacts()
        )

    def _advance_until_inserted_stable(self) -> bool:
        """Wait for three stable insertion checks, bounded by simulation time."""
        check_interval = 25
        minimum_steps = SETTLE_SIMULATION_STEPS
        stable_checks = 0
        elapsed_steps = 0
        while elapsed_steps < DIRECT_RELEASE_SETTLE_MAX_STEPS:
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

    def _advance_until_motion_settled(self, stage: str) -> bool:
        """Advance physics until the arm and held object are stably near rest."""
        rigid_body = next(
            component
            for component in self.cylinder.actor.get_components()
            if hasattr(component, "get_linear_velocity")
        )
        qpos_window: list[np.ndarray] = []
        minimum_arm_speed = math.inf
        minimum_linear_speed = math.inf
        minimum_angular_speed = math.inf
        arm_speed = linear_speed = angular_speed = math.inf
        joint_position_span = math.inf
        used_steps = 0
        settled = False
        for used_steps in range(1, MOTION_SETTLE_MAX_STEPS + 1):
            arm_speed = float(
                np.linalg.norm(np.asarray(self.robot.left_entity.get_qvel())[:6])
            )
            linear_speed = float(np.linalg.norm(rigid_body.get_linear_velocity()))
            angular_speed = float(np.linalg.norm(rigid_body.get_angular_velocity()))
            minimum_arm_speed = min(minimum_arm_speed, arm_speed)
            minimum_linear_speed = min(minimum_linear_speed, linear_speed)
            minimum_angular_speed = min(minimum_angular_speed, angular_speed)
            qpos_window.append(
                np.asarray(self.robot.left_entity.get_qpos(), dtype=float)[:6]
            )
            if len(qpos_window) > MOTION_SETTLE_WINDOW_STEPS:
                qpos_window.pop(0)
            if len(qpos_window) == MOTION_SETTLE_WINDOW_STEPS:
                joint_position_span = float(
                    np.max(np.ptp(np.vstack(qpos_window), axis=0))
                )
                if (
                    joint_position_span < MOTION_SETTLE_MAX_JOINT_SPAN_RAD
                    and linear_speed < 0.02
                    and angular_speed < 0.10
                ):
                    settled = True
                    break
            self._advance_physics(1)
        if not hasattr(self, "motion_settle_audit"):
            self.motion_settle_audit = []
        self.motion_settle_audit.append(
            {
                "stage": stage,
                "settled": settled,
                "simulation_steps": used_steps,
                "final_arm_speed_radps": arm_speed,
                "final_object_linear_speed_mps": linear_speed,
                "final_object_angular_speed_radps": angular_speed,
                "minimum_arm_speed_radps": minimum_arm_speed,
                "minimum_object_linear_speed_mps": minimum_linear_speed,
                "minimum_object_angular_speed_radps": minimum_angular_speed,
                "joint_position_window_steps": len(qpos_window),
                "joint_position_span_rad": joint_position_span,
            }
        )
        return settled

    def _carry_object_center_to(
        self,
        target_position: np.ndarray,
        stage: str,
        tolerance_m: float = 0.012,
        target_quaternion_wxyz: list[float] | None = None,
        max_segment_m: float | None = None,
    ) -> bool:
        """Carry through retimed waypoints without stopping at each waypoint."""
        target = np.asarray(target_position, dtype=float)
        object_start = np.asarray(self.cylinder.get_pose().p, dtype=float)
        displacement = target - object_start
        distance = float(np.linalg.norm(displacement))
        if max_segment_m is None:
            waypoint_count = 1
        else:
            if max_segment_m <= 0.0:
                raise ValueError("max_segment_m must be positive")
            waypoint_count = max(1, int(math.ceil(distance / max_segment_m)))
        ee_start = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        quaternion = (
            ee_start[3:].tolist()
            if target_quaternion_wxyz is None
            else _normalize_quaternion(
                np.asarray(target_quaternion_wxyz, dtype=float)
            ).tolist()
        )
        target_poses = [
            (
                ee_start[:3] + displacement * (index / waypoint_count)
            ).tolist()
            + quaternion
            for index in range(1, waypoint_count + 1)
        ]
        if not self._execute_continuous_ee_waypoints(target_poses, stage):
            return False
        if not self._advance_until_motion_settled(stage):
            self.capture_oracle_stage(f"{stage}_settle_timeout")
            return False
        final_error = float(
            np.linalg.norm(target - np.asarray(self.cylinder.get_pose().p, dtype=float))
        )
        return final_error <= tolerance_m

    def _sample_task_scene(self, rng: np.random.Generator) -> dict:
        """Map the deterministic v2 sampling contract to the simulator scene."""
        del rng
        workspace_values = self.task_randomization.get("workspace", {}) or {}
        sampled = sample_scene(self.episode_seed, workspace_values)
        forced_posture = self.task_randomization.get("forced_posture")
        if forced_posture is not None:
            if forced_posture not in {"upright", "lying"}:
                raise ValueError("forced_posture must be 'upright' or 'lying'")
            sampled["cylinder_posture"] = forced_posture
        forced_angle_bin = os.environ.get("PANTHERA_V2_FORCED_LYING_ANGLE_BIN")
        if forced_angle_bin is None:
            forced_angle_bin = self.task_randomization.get("forced_lying_angle_bin")
        if forced_angle_bin is not None:
            forced_angle_bin = int(forced_angle_bin)
            if not 0 <= forced_angle_bin < 8:
                raise ValueError("forced_lying_angle_bin must be in [0, 7]")
            if sampled["cylinder_posture"] != "lying":
                raise ValueError("a forced lying angle requires forced_posture: lying")
            angle_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self.episode_seed, forced_angle_bin, 0x4C59494E]
                )
            )
            low = forced_angle_bin * math.pi / 8.0
            high = (forced_angle_bin + 1) * math.pi / 8.0
            sampled["lying_angle_bin"] = forced_angle_bin
            sampled["cylinder_angle_rad"] = float(angle_rng.uniform(low, high))
        elif sampled["cylinder_posture"] == "upright":
            sampled["lying_angle_bin"] = None
            sampled["cylinder_angle_rad"] = 0.0
        angle = float(sampled["cylinder_angle_rad"])
        if sampled["cylinder_posture"] == "upright":
            center_z = TABLE_HEIGHT_M + CYLINDER_HALF_HEIGHT_M + 0.002
            quaternion = [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
            axis = [0.0, 0.0, 1.0]
        else:
            center_z = TABLE_HEIGHT_M + CYLINDER_RADIUS_M + 0.002
            quaternion = [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]
            axis = [math.cos(angle), math.sin(angle), 0.0]
        return {
            **sampled,
            "groove_target_xy_m": sampled["socket_target_xy_m"],
            "cylinder_center_z_m": center_z,
            "cylinder_quaternion_wxyz": quaternion,
            "cylinder_axis_world": axis,
        }

    def _configure_grasp_for_scene(self, scene_sample: dict):
        """Initialize the preferred cylinder end; planning selects the route."""
        self.lying_grasp_physical_axis_sign = 1.0
        self.reorientation_radial_mode = "base"
        if scene_sample["cylinder_posture"] == "upright":
            quaternion = TOP_DOWN_GRASP_QUATERNION_WXYZ
        else:
            angle = float(scene_sample["cylinder_angle_rad"])
            physical_axis = np.array([math.cos(angle), math.sin(angle)])
            workspace = scene_sample["workspace"]
            robot_base = np.array(
                [workspace["robot_base_x_m"], workspace["robot_base_y_m"]],
                dtype=float,
            )
            cylinder_xy = np.asarray(
                scene_sample["cylinder_initial_xy_m"], dtype=float
            )
            toward_robot = robot_base - cylinder_xy
            if float(np.dot(physical_axis, toward_robot)) < 0.0:
                self.lying_grasp_physical_axis_sign = -1.0
            quaternion = _lying_grasp_quaternion(
                angle
                + (
                    math.pi
                    if self.lying_grasp_physical_axis_sign < 0.0
                    else 0.0
                )
            )
        self.grasp_quaternion_wxyz = _normalize_quaternion(
            np.asarray(quaternion, dtype=float)
        ).tolist()
        rotation = t3d.quaternions.quat2mat(self.grasp_quaternion_wxyz)
        self.grasp_approach_axis_world = rotation[:, 0].tolist()
        self.finger_closing_axis_world = rotation[:, 1].tolist()

    def _apply_lying_grasp_candidate(
        self,
        physical_axis_sign: float,
        radial_mode: str,
    ):
        """Configure one of the finite preflight grasp-route candidates."""
        if radial_mode not in {"base", "side"}:
            raise ValueError("unknown reorientation radial mode")
        angle = float(self.realized_geometry["cylinder_angle_rad"])
        self.lying_grasp_physical_axis_sign = math.copysign(
            1.0, physical_axis_sign
        )
        self.reorientation_radial_mode = radial_mode
        directed_angle = angle + (
            math.pi if self.lying_grasp_physical_axis_sign < 0.0 else 0.0
        )
        quaternion = _lying_grasp_quaternion(directed_angle)
        self.grasp_quaternion_wxyz = quaternion
        rotation = t3d.quaternions.quat2mat(quaternion)
        self.grasp_approach_axis_world = rotation[:, 0].tolist()
        self.finger_closing_axis_world = rotation[:, 1].tolist()

    def _grasp_target_poses(self) -> tuple[list[float], list[float]]:
        """Return pregrasp and grasp poses for the active candidate."""
        point = np.asarray(self.cylinder.get_contact_point(0, "pose").p, dtype=float)
        if self.cylinder_posture == "lying":
            offset = float(
                os.environ.get(
                    "PANTHERA_V2_LYING_GRASP_AXIS_OFFSET_M",
                    str(LYING_END_GRASP_OFFSET_M),
                )
            )
            if not 0.0 < offset <= LYING_END_GRASP_MAX_OFFSET_M:
                raise ValueError("lying grasp offset is outside the cylinder")
            physical_axis = t3d.quaternions.quat2mat(self.cylinder.get_pose().q)[:, 0]
            point = point + physical_axis * (
                offset * self.lying_grasp_physical_axis_sign
            )
        quaternion = np.asarray(self.grasp_quaternion_wxyz, dtype=float)
        rotation = t3d.quaternions.quat2mat(quaternion)
        pregrasp = point + rotation @ np.array([-0.15, 0.0, 0.0])
        grasp = point + rotation @ np.array([-0.12, 0.0, 0.0])
        return (
            pregrasp.tolist() + quaternion.tolist(),
            grasp.tolist() + quaternion.tolist(),
        )

    def _top_down_grasp(self, arm: ArmTag, contact_point_id: int):
        """Approach vertically and grasp a lying cylinder near its upper end.

        A centered grasp leaves the gripper beside the cylinder midpoint after
        the cylinder is rotated upright.  Offsetting the contact point along
        the cylinder axis keeps the same physical grasp while placing the
        gripper near the upper end, so the object can be carried directly to
        the socket without being released and grasped a second time.
        """
        if contact_point_id != 0:
            raise ValueError("randomized cylinder has one grasp contact point")
        self.lying_grasp_axis_offset_m = 0.0
        self.lying_grasp_physical_axis_offset_m = 0.0
        if self.cylinder_posture == "lying":
            offset = float(
                os.environ.get(
                    "PANTHERA_V2_LYING_GRASP_AXIS_OFFSET_M",
                    str(LYING_END_GRASP_OFFSET_M),
                )
            )
            if abs(offset) > LYING_END_GRASP_MAX_OFFSET_M:
                raise ValueError("lying grasp offset is outside the cylinder")
            physical_offset = offset * self.lying_grasp_physical_axis_sign
            self.lying_grasp_axis_offset_m = offset
            self.lying_grasp_physical_axis_offset_m = physical_offset
        pregrasp, grasp = self._grasp_target_poses()
        return arm, [
            Action(arm, "move", target_pose=pregrasp),
            Action(arm, "move", target_pose=grasp),
            Action(arm, "close", target_gripper_pos=V2_GRASP_OPENING),
        ]

    def _build_single_grasp_route_targets(
        self,
        object_center: np.ndarray,
        start_quaternion: np.ndarray,
        local_object_offset: np.ndarray,
    ) -> tuple[list[list[float]], np.ndarray, float]:
        """Build the complete held-object route without moving the simulator."""
        planned_center = np.asarray(object_center, dtype=float).copy()
        start_quaternion = _normalize_quaternion(start_quaternion)
        start_rotation = t3d.quaternions.quat2mat(start_quaternion)
        local_object_offset = np.asarray(local_object_offset, dtype=float)
        target_poses: list[list[float]] = []

        def object_to_ee(center: np.ndarray, quaternion: np.ndarray) -> list[float]:
            rotation = t3d.quaternions.quat2mat(quaternion)
            target_ee = center - rotation @ local_object_offset
            return target_ee.tolist() + quaternion.tolist()

        def append_translation(target_center: np.ndarray, quaternion: np.ndarray):
            nonlocal planned_center
            target = np.asarray(target_center, dtype=float)
            if float(np.linalg.norm(target - planned_center)) > 1.0e-6:
                target_poses.append(object_to_ee(target, quaternion))
            planned_center = target

        append_translation(
            np.array([planned_center[0], planned_center[1], SAFE_CARRY_Z_M]),
            start_quaternion,
        )
        carry_quaternion = start_quaternion
        socket_center = np.asarray(self.groove_target_pose.p, dtype=float)
        if self.cylinder_posture == "lying":
            workspace = self.realized_geometry["workspace"]
            robot_base = np.asarray(
                [workspace["robot_base_x_m"], workspace["robot_base_y_m"]],
                dtype=float,
            )
            station_xy = _mirrored_station(
                float(robot_base[0]),
                float(planned_center[0]),
                REORIENTATION_STATION_XY_M,
            )
            station_xy = (
                station_xy
                - start_rotation[:2, 2] * self.lying_grasp_axis_offset_m
            )
            append_translation(
                np.array([station_xy[0], station_xy[1], SAFE_CARRY_Z_M]),
                start_quaternion,
            )
            if self.reorientation_radial_mode == "base":
                radial = planned_center[:2] - robot_base
            elif self.reorientation_radial_mode == "side":
                side = -1.0 if planned_center[0] < robot_base[0] else 1.0
                radial = np.array([side, 0.0], dtype=float)
            else:
                raise ValueError("unknown reorientation radial mode")
            carry_quaternion = np.asarray(
                _upright_side_grasp_quaternion(radial), dtype=float
            )
            for index in range(1, REORIENTATION_STEPS + 1):
                quaternion = _slerp_wxyz(
                    start_quaternion,
                    carry_quaternion,
                    index / REORIENTATION_STEPS,
                )
                target_poses.append(object_to_ee(planned_center, quaternion))
            socket_radial = socket_center[:2] - robot_base
            if np.linalg.norm(socket_radial) <= 1.0e-8:
                raise ValueError("socket radial direction is undefined")
            carry_quaternion = np.asarray(
                _upright_side_grasp_quaternion(socket_radial), dtype=float
            )
        append_translation(
            np.array([socket_center[0], socket_center[1], SAFE_CARRY_Z_M]),
            carry_quaternion,
        )
        release_clearance = float(
            os.environ.get(
                "PANTHERA_V2_DIRECT_RELEASE_BOTTOM_CLEARANCE_M",
                str(DIRECT_RELEASE_BOTTOM_CLEARANCE_M),
            )
        )
        if not DIRECT_RELEASE_MIN_CLEARANCE_M <= release_clearance <= DIRECT_RELEASE_MAX_CLEARANCE_M:
            raise ValueError("direct-release clearance is outside the search range")
        release_center_z = (
            float(self.socket_top_z) + CYLINDER_HALF_HEIGHT_M + release_clearance
        )
        append_translation(
            np.array([socket_center[0], socket_center[1], release_center_z]),
            carry_quaternion,
        )
        return target_poses, carry_quaternion, release_clearance

    def _pose_sequence_is_plannable(
        self,
        target_poses: list[list[float]],
        last_qpos: np.ndarray | None = None,
    ) -> tuple[bool, np.ndarray | None]:
        """Check a pose sequence without executing it or changing task status."""
        current = last_qpos
        for target_pose in target_poses:
            try:
                result = self.robot.left_plan_path(target_pose, last_qpos=current)
            except Exception:
                return False, None
            if result is None or result.get("status") != "Success":
                return False, None
            current = np.asarray(result["position"][-1], dtype=float)
        return True, current

    def _select_lying_grasp_route(self) -> bool:
        """Choose a complete feasible route before any arm motion begins."""
        if self.cylinder_posture != "lying":
            return True
        object_center = np.asarray(self.cylinder.get_pose().p, dtype=float)
        workspace = self.realized_geometry["workspace"]
        robot_base = np.asarray(
            [workspace["robot_base_x_m"], workspace["robot_base_y_m"]],
            dtype=float,
        )
        physical_axis = t3d.quaternions.quat2mat(self.cylinder.get_pose().q)[:, 0]
        nearest_sign = (
            1.0
            if float(np.dot(physical_axis[:2], robot_base - object_center[:2])) >= 0.0
            else -1.0
        )
        offset = float(
            os.environ.get(
                "PANTHERA_V2_LYING_GRASP_AXIS_OFFSET_M",
                str(LYING_END_GRASP_OFFSET_M),
            )
        )
        if not 0.0 < offset <= LYING_END_GRASP_MAX_OFFSET_M:
            raise ValueError("lying grasp offset is outside the cylinder")
        self.lying_grasp_axis_offset_m = offset
        forced_sign_value = os.environ.get("PANTHERA_V2_GRASP_AXIS_SIGN")
        signs = (
            (math.copysign(1.0, float(forced_sign_value)),)
            if forced_sign_value is not None
            else (nearest_sign, -nearest_sign)
        )
        forced_radial_mode = os.environ.get("PANTHERA_V2_REORIENTATION_RADIAL_MODE")
        if forced_radial_mode is not None and forced_radial_mode not in {"base", "side"}:
            raise ValueError("forced reorientation radial mode must be base or side")
        radial_modes = (
            (forced_radial_mode,)
            if forced_radial_mode is not None
            else ("base", "side")
        )
        audits: list[dict[str, object]] = []
        for physical_sign in signs:
            for radial_mode in radial_modes:
                self._apply_lying_grasp_candidate(physical_sign, radial_mode)
                pregrasp, grasp = self._grasp_target_poses()
                grasp_ok, grasp_qpos = self._pose_sequence_is_plannable(
                    [pregrasp, grasp]
                )
                route_ok = False
                if grasp_ok and grasp_qpos is not None:
                    route_targets, _, _ = self._build_single_grasp_route_targets(
                        object_center,
                        np.asarray(self.grasp_quaternion_wxyz, dtype=float),
                        np.array([0.12, 0.0, -offset], dtype=float),
                    )
                    route_ok, _ = self._pose_sequence_is_plannable(
                        route_targets, grasp_qpos
                    )
                audits.append(
                    {
                        "physical_axis_sign": physical_sign,
                        "radial_mode": radial_mode,
                        "grasp_plannable": grasp_ok,
                        "route_plannable": route_ok,
                    }
                )
                if grasp_ok and route_ok:
                    self.grasp_route_selection_audit = audits
                    return True
        self.grasp_route_selection_audit = audits
        self.capture_oracle_stage("no_complete_single_grasp_route")
        return False

    def _execute_single_grasp_route(self) -> bool:
        """Lift, reorient, carry and lower in one retimed held-object motion."""
        object_center = np.asarray(self.cylinder.get_pose().p, dtype=float)
        ee_pose = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        start_quaternion = _normalize_quaternion(ee_pose[3:])
        start_rotation = t3d.quaternions.quat2mat(start_quaternion)
        local_object_offset = start_rotation.T @ (object_center - ee_pose[:3])
        offset_norm = float(np.linalg.norm(local_object_offset))
        if not 0.07 <= offset_norm <= 0.16:
            self.capture_oracle_stage("single_grasp_offset_rejected")
            return False
        target_poses, carry_quaternion, release_clearance = (
            self._build_single_grasp_route_targets(
                object_center, start_quaternion, local_object_offset
            )
        )
        if not self._execute_continuous_ee_waypoints(
            target_poses, "single_grasp_route_ready_to_release"
        ):
            return False
        if not self._advance_until_motion_settled("single_grasp_route_ready_to_release"):
            self.capture_oracle_stage("single_grasp_release_settle_timeout")
            return False
        self.direct_release_bottom_clearance_m = release_clearance
        self.carry_quaternion_wxyz = carry_quaternion.tolist()
        self.capture_oracle_stage("direct_release_ready")
        if self._has_gripper_socket_contact():
            self.capture_oracle_stage("direct_release_gripper_socket_contact")
            return False
        return True

    def _reorient_lying_cylinder(self) -> bool:
        """Rotate the physically held cylinder upright without teleporting it."""
        object_center = np.asarray(self.cylinder.get_pose().p, dtype=float)
        object_center[2] = SAFE_CARRY_Z_M
        ee_pose = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        start_quaternion = _normalize_quaternion(ee_pose[3:])
        start_rotation = t3d.quaternions.quat2mat(start_quaternion)
        local_object_offset = start_rotation.T @ (
            np.asarray(self.cylinder.get_pose().p, dtype=float) - ee_pose[:3]
        )
        if not 0.07 <= float(np.linalg.norm(local_object_offset)) <= 0.16:
            self.capture_oracle_stage("reorientation_offset_rejected")
            return False

        workspace = self.realized_geometry["workspace"]
        robot_base = np.asarray(
            [workspace["robot_base_x_m"], workspace["robot_base_y_m"]],
            dtype=float,
        )
        side = -1.0 if object_center[0] < robot_base[0] else 1.0
        radial = np.array([side, 0.0], dtype=float)
        configured_yaw = os.environ.get("PANTHERA_V2_SIDE_GRASP_YAW_DEG")
        if configured_yaw is not None:
            yaw = math.radians(float(configured_yaw))
            radial = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
        target_quaternion = np.asarray(
            _upright_side_grasp_quaternion(radial), dtype=float
        )
        target_poses = []
        for index in range(1, REORIENTATION_STEPS + 1):
            quaternion = _slerp_wxyz(
                start_quaternion, target_quaternion, index / REORIENTATION_STEPS
            )
            rotation = t3d.quaternions.quat2mat(quaternion)
            target_ee = object_center - rotation @ local_object_offset
            target_poses.append(target_ee.tolist() + quaternion.tolist())
        if not self._execute_continuous_ee_waypoints(
            target_poses, "reoriented_continuously"
        ):
            return False
        if not self._advance_until_motion_settled("reoriented_continuously"):
            self.capture_oracle_stage("reorientation_settle_timeout")
            return False
        self.carry_quaternion_wxyz = target_quaternion.tolist()
        self.release_withdrawal_xy = -radial / np.linalg.norm(radial) * 0.08
        return True

    def _release_and_regrasp_upright(self) -> bool:
        """Set the reoriented cylinder down, then acquire the proven top grasp."""
        arm = ArmTag("left")
        release_z = float(
            os.environ.get(
                "PANTHERA_V2_REORIENTATION_RELEASE_Z_M",
                str(REORIENTATION_RELEASE_Z_M),
            )
        )
        if not MIN_REORIENTATION_RELEASE_Z_M <= release_z <= SAFE_CARRY_Z_M:
            raise ValueError("reorientation release height is outside the safe range")
        workspace = self.realized_geometry["workspace"]
        robot_base_x = float(workspace["robot_base_x_m"])
        object_x = float(self.cylinder.get_pose().p[0])
        regrasp_xy = _mirrored_station(
            robot_base_x, object_x, REGRASP_STATION_XY_M
        )
        high_lift_xy = _mirrored_station(
            robot_base_x, object_x, HIGH_LIFT_STATION_XY_M
        )
        station = np.array(
            [
                regrasp_xy[0],
                regrasp_xy[1],
                release_z,
            ]
        )
        if not self._carry_object_center_to(
            np.array([station[0], station[1], SAFE_CARRY_Z_M]),
            "moved_upright_to_regrasp_station",
            tolerance_m=0.030,
            target_quaternion_wxyz=self.carry_quaternion_wxyz,
            max_segment_m=0.025,
        ):
            return False
        if not self._carry_object_center_to(
            station,
            "lowered_upright_at_station",
            tolerance_m=0.030,
            target_quaternion_wxyz=self.carry_quaternion_wxyz,
            max_segment_m=0.020,
        ):
            return False
        if not self._move_arm(
            self.open_gripper(arm), "released_upright_at_station"
        ):
            return False
        self._advance_physics(SETTLE_SIMULATION_STEPS)
        self.capture_oracle_stage("upright_station_settled")
        axis = t3d.quaternions.quat2mat(self.cylinder.get_pose().q)[:, 0]
        if abs(float(axis[2])) < math.cos(math.radians(15.0)):
            self.capture_oracle_stage("upright_station_rejected")
            return False
        lift_clearance = SAFE_CARRY_Z_M - release_z
        if lift_clearance > 0.005:
            if not self._move_arm(
                self.move_by_displacement(
                    arm,
                    x=0.0,
                    y=0.0,
                    z=lift_clearance,
                    quat=self.carry_quaternion_wxyz,
                ),
                "side_gripper_lifted_clear",
            ):
                return False
        if not self._move_arm(
            self.back_to_origin(arm), "side_gripper_retracted"
        ):
            return False

        quaternion = _normalize_quaternion(
            np.asarray(TOP_DOWN_GRASP_QUATERNION_WXYZ, dtype=float)
        )
        self.grasp_quaternion_wxyz = quaternion.tolist()
        rotation = t3d.quaternions.quat2mat(quaternion)
        self.grasp_approach_axis_world = rotation[:, 0].tolist()
        self.finger_closing_axis_world = rotation[:, 1].tolist()
        if not self._move_arm(self._top_down_grasp(arm, 0), "regrasped_from_top"):
            return False
        self._advance_physics(25)
        self.capture_oracle_stage("top_regrasp_settled")
        if len(self.get_gripper_actor_contact_position("panthera_cylinder")) < 2:
            self.capture_oracle_stage("top_regrasp_contact_rejected")
            return False
        current = np.asarray(self.cylinder.get_pose().p, dtype=float)
        if not self._carry_object_center_to(
            np.array([current[0], current[1], REGRASP_TRANSFER_Z_M]),
            "relifted_for_transfer",
            max_segment_m=0.030,
        ):
            return False
        if not self._carry_object_center_to(
            np.array(
                [
                    high_lift_xy[0],
                    high_lift_xy[1],
                    REGRASP_TRANSFER_Z_M,
                ]
            ),
            "moved_to_high_lift_station",
            max_segment_m=0.025,
        ):
            return False
        if not self._carry_object_center_to(
            np.array(
                [
                    high_lift_xy[0],
                    high_lift_xy[1],
                    SAFE_CARRY_Z_M,
                ]
            ),
            "relifted_after_regrasp",
            max_segment_m=0.025,
        ):
            return False
        self.carry_quaternion_wxyz = quaternion.tolist()
        return True

    def _carry_to_socket(self) -> bool:
        """Carry at clearance height, then descend to a bounded release pose."""
        target = np.asarray(self.groove_target_pose.p, dtype=float)
        quaternion = self.carry_quaternion_wxyz
        waypoints = [
            np.array([target[0], target[1], SAFE_CARRY_Z_M]),
            target + np.array([0.0, 0.0, 0.045]),
        ]
        for index, waypoint in enumerate(waypoints, start=1):
            if not self._carry_object_center_to(
                waypoint,
                f"socket_approach_{index}",
                tolerance_m=0.012,
                target_quaternion_wxyz=quaternion,
                max_segment_m=0.035,
            ):
                return False
        return True

    def play_once(self):
        """Grasp, optionally reorient, transport, release, settle and retract."""
        arm = ArmTag("left")
        if not self._move_arm(
            self.close_gripper(arm, pos=PREGRASP_OPENING), "preclose"
        ):
            return self.info
        if not self._select_lying_grasp_route():
            return self.info
        if not self._move_arm(self._top_down_grasp(arm, 0), "grasped"):
            return self.info
        self._advance_physics(25)
        self.capture_oracle_stage("grasp_settled")
        if len(self.get_gripper_actor_contact_position("panthera_cylinder")) < 2:
            self.capture_oracle_stage("grasp_contact_rejected")
            return self.info

        if not self._execute_single_grasp_route():
            return self.info
        self._configure_release_contact_solver()
        if not self._move_arm(self.open_gripper(arm), "released_above_socket"):
            return self.info
        self._advance_physics(25)
        self.capture_oracle_stage("release_settled")

        if not self._move_arm(self.back_to_origin(arm), "retracted"):
            return self.info

        self._advance_until_inserted_stable()
        self.capture_oracle_stage("inserted_and_settled")
        self._finalize_episode_info()
        return self.info

    def get_info(self):
        return {"{A}": "upright or lying cylinder", "{B}": "yellow socket"}

    def _finalize_episode_info(self):
        """Add the globally retimed motion contract to the episode metadata."""
        super()._finalize_episode_info()
        episode = self.info["panthera_episode"]
        episode["motion_profile"] = "chaikin_arc_length_quintic_single_grasp_v3"
        episode["continuous_speed_scale"] = CONTINUOUS_SPEED_SCALE
        episode["continuous_motion_audit"] = list(
            getattr(self, "continuous_motion_audit", [])
        )
        episode["motion_settle_audit"] = list(
            getattr(self, "motion_settle_audit", [])
        )
        episode["joint_smoothing_audit"] = list(
            getattr(self, "joint_smoothing_audit", [])
        )
        episode["joint_retime_audit"] = list(
            getattr(self, "joint_retime_audit", [])
        )
        episode["single_grasp"] = True
        episode["direct_release_bottom_clearance_m"] = float(
            self.direct_release_bottom_clearance_m
        )
        episode["lying_grasp_axis_offset_m"] = float(
            getattr(self, "lying_grasp_axis_offset_m", 0.0)
        )
        episode["lying_grasp_physical_axis_offset_m"] = float(
            getattr(self, "lying_grasp_physical_axis_offset_m", 0.0)
        )
        episode["reorientation_radial_mode"] = str(
            getattr(self, "reorientation_radial_mode", "base")
        )
        episode["grasp_route_selection_audit"] = list(
            getattr(self, "grasp_route_selection_audit", [])
        )
        episode["direct_release_lowest_validated_clearance_m"] = (
            DIRECT_RELEASE_LOWEST_VALIDATED_CLEARANCE_M
        )
        episode["direct_release_clearance_margin_m"] = (
            DIRECT_RELEASE_CLEARANCE_MARGIN_M
        )
