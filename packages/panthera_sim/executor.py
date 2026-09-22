"""The one executor a 50 Hz action stream is run through.

The expert replay and the policy rollout have to be the same executor or neither
number means anything: the replay exists to establish the ceiling the policy is
measured against.  They were two implementations, and the differences were not
cosmetic -- one applied the release contact solver when the gripper reopened,
the other only at the end, which decides whether a correctly seated cylinder
ever comes to rest.

Behaviour here is the configuration that produced the recorded baseline of
1249/1280 expert replays: finite-difference velocity feedforward, no speed
clamp, next-sample action alignment, release solver on reopening, and a final
settle.  The other knobs exist because they were tested and rejected, and the
reasons are recorded next to them so they are not tried again blind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

SAMPLE_PERIOD_STEPS = 5
PHYSICS_DT = 1.0 / 250.0
# The oracle holds at 0.3 and pregrasps at 0.9, so below the midpoint is
# unambiguously a closed gripper.
GRIPPER_CLOSED_OPENING = 0.5
VELOCITY_MODES = ("zero", "finite_difference")


@dataclass(frozen=True)
class ExecutorConfig:
    """How a 50 Hz target stream is turned into physics steps."""

    velocity_mode: str = "finite_difference"
    # A position drive lags in proportion to speed over stiffness, and the
    # feedforward cancels it.  Measured: zero feedforward leaves 27.7 mrad of
    # median deviation from the recording, finite difference leaves 4.6.  Gains
    # above 1.0 were swept and are not monotonic -- 1.5 fixes one episode and
    # breaks another -- so 1.0 stands.
    velocity_gain: float = 1.0
    # Effectively disabled.  The limiter carries its own state, so once it falls
    # behind it never catches up, and it silently changes the trajectory.
    speed_limit_radps: float = 1000.0
    # Ramping between grid targets reconstructs the 250 Hz command stream more
    # closely, but measured on the full set it trades episodes rather than
    # winning: 36/40 against 39/40.
    interpolate: bool = False
    # Holding each target until the arm arrives removes the standing lag, but it
    # stretches the recorded timing, and the lying route depends on that timing
    # to keep the object in the gripper: 23/40 at a 5 mrad tolerance.
    converge_tolerance_rad: float = 0.0
    converge_max_steps: int = 25
    final_settle: bool = True

    def __post_init__(self) -> None:
        if self.velocity_mode not in VELOCITY_MODES:
            raise ValueError(f"velocity_mode must be one of {VELOCITY_MODES}")
        if self.velocity_gain <= 0.0:
            raise ValueError("velocity_gain must be positive")
        if self.speed_limit_radps <= 0.0:
            raise ValueError("speed_limit_radps must be positive")
        if self.converge_tolerance_rad < 0.0:
            raise ValueError("converge_tolerance_rad must be non-negative")
        if self.converge_max_steps < SAMPLE_PERIOD_STEPS:
            raise ValueError("converge_max_steps cannot be below the grid period")


class _Progress:
    """Track how far along the task chain an episode got.

    The nine success criteria are continuous, but they saturate: an episode that
    never touches the cylinder ends it exactly where it started, so every
    distance reads its initial value no matter what the arm did.  These
    quantities move before success does, which is what makes a failing run
    comparable to another failing run.
    """

    def __init__(self, task, reference: Optional[np.ndarray] = None) -> None:
        self.task = task
        # The expert's measured joint angles, when the caller has them.  This is
        # the most sensitive signal available while the policy never reaches the
        # object: two runs that both score zero still differ in how long they
        # stayed on the trajectory that works.
        self.reference = None if reference is None else np.asarray(reference, dtype=np.float64)
        self.deviations: list[float] = []
        pose = task.cylinder.get_pose()
        self.start = np.asarray(pose.p, dtype=np.float64)
        geometry = task.realized_geometry
        target = geometry.get("socket_target_xy_m") or geometry["groove_target_xy_m"]
        self.socket_xy = np.asarray(target, dtype=np.float64)
        self.nearest_gripper_m = float("inf")
        self.max_lift_m = 0.0
        self.max_travel_m = 0.0
        self.nearest_socket_m = float(
            np.linalg.norm(self.start[:2] - self.socket_xy)
        )
        self.initial_socket_m = self.nearest_socket_m
        self.contact_steps = 0
        self.grasp_steps = 0
        # A stage that was reached and then lost is a different failure from one
        # never reached: an episode that grasps, lifts, carries and then releases
        # too high reports the same furthest stage as one still holding the
        # object at the end.  First-reached and still-true separate them.
        self.first_reached: dict[int, int] = {}
        self.last_true: dict[int, int] = {}
        self.step = 0

    def observe(self, step: int = -1) -> None:
        if self.reference is not None and 0 <= step < len(self.reference):
            measured = np.asarray(
                self.task._actual_robot_state()["arm_qpos"], dtype=np.float64
            )
            self.deviations.append(
                float(np.abs(measured - self.reference[step][:6]).max())
            )
        pose = np.asarray(self.task.cylinder.get_pose().p, dtype=np.float64)
        try:
            ee = np.asarray(self.task.robot.get_left_ee_pose(), dtype=np.float64)[:3]
            self.nearest_gripper_m = min(
                self.nearest_gripper_m, float(np.linalg.norm(ee - pose))
            )
        except Exception:
            pass
        self.max_lift_m = max(self.max_lift_m, float(pose[2] - self.start[2]))
        self.max_travel_m = max(
            self.max_travel_m, float(np.linalg.norm(pose - self.start))
        )
        self.nearest_socket_m = min(
            self.nearest_socket_m, float(np.linalg.norm(pose[:2] - self.socket_xy))
        )
        contacts = len(
            self.task.get_gripper_actor_contact_position("panthera_cylinder")
        )
        if contacts >= 1:
            self.contact_steps += 1
        if contacts >= 2:
            self.grasp_steps += 1

        self.step = max(self.step, step)
        socket_now = float(np.linalg.norm(pose[:2] - self.socket_xy))
        lift_now = float(pose[2] - self.start[2])
        holding = contacts >= 2
        for level, satisfied in (
            (1, self._gripper_near(pose)),
            (2, holding),
            (3, lift_now > 0.02),
            (4, socket_now < self.initial_socket_m - 0.05),
            (5, socket_now <= 0.010),
        ):
            if satisfied:
                self.first_reached.setdefault(level, step)
                self.last_true[level] = step

    def _gripper_near(self, pose: np.ndarray) -> bool:
        try:
            ee = np.asarray(self.task.robot.get_left_ee_pose(), dtype=np.float64)[:3]
        except Exception:
            return False
        return bool(np.linalg.norm(ee - pose) < 0.10)

    STAGE_NAMES = {
        0: "从未靠近", 1: "接近", 2: "接触夹持", 3: "抬起", 4: "搬运到槽附近", 5: "插入",
    }

    def stage(self) -> int:
        """The furthest stage reached at any point; 0 never approached."""
        return max(self.first_reached) if self.first_reached else 0

    def final_stage(self) -> int:
        """The highest stage still satisfied at the end of the episode."""
        held = [level for level, last in self.last_true.items() if last >= self.step]
        return max(held) if held else 0

    def failure(self) -> dict:
        """Where the episode stopped making progress, and whether it regressed.

        ``reached`` is the furthest stage; ``final`` is what was still true at
        the end.  ``final < reached`` means the episode lost something it had --
        a dropped object, a cylinder released too high and toppled -- which is a
        different defect from never getting there.
        """
        reached = self.stage()
        final = self.final_stage()
        return {
            "reached_stage": reached,
            "reached_name": self.STAGE_NAMES[reached],
            "final_stage": final,
            "final_name": self.STAGE_NAMES[final],
            "regressed": final < reached,
            "failed_at": self.STAGE_NAMES[min(reached + 1, 5)] if reached < 5 else None,
            "first_reached_step": dict(sorted(self.first_reached.items())),
            "last_true_step": dict(sorted(self.last_true.items())),
        }

    def deviation_report(self) -> Optional[dict]:
        if not self.deviations:
            return None
        values = np.asarray(self.deviations, dtype=np.float64)
        exceeded = np.flatnonzero(values > 0.1)
        return {
            "median_mrad": round(float(np.median(values)) * 1000, 2),
            "p99_mrad": round(float(np.percentile(values, 99)) * 1000, 2),
            "steps_compared": int(len(values)),
            # How long the policy stayed on the path that is known to work.
            "first_step_over_100mrad": int(exceeded[0]) if len(exceeded) else None,
        }

    def report(self) -> dict:
        return {
            "stage": self.stage(),
            "failure": self.failure(),
            "expert_deviation": self.deviation_report(),
            "nearest_gripper_to_object_m": (
                None if self.nearest_gripper_m == float("inf")
                else round(self.nearest_gripper_m, 5)
            ),
            "contact_steps": self.contact_steps,
            "grasp_steps": self.grasp_steps,
            "max_lift_m": round(self.max_lift_m, 5),
            "max_object_travel_m": round(self.max_travel_m, 5),
            "initial_object_to_socket_m": round(self.initial_socket_m, 5),
            "nearest_object_to_socket_m": round(self.nearest_socket_m, 5),
            "closed_fraction": (
                round(1.0 - self.nearest_socket_m / self.initial_socket_m, 4)
                if self.initial_socket_m > 0 else None
            ),
        }


@dataclass
class DenseExecutor:
    """Drive one task with 50 Hz targets, one grid period at a time."""

    task: Any
    config: ExecutorConfig = field(default_factory=ExecutorConfig)
    ratchet: Optional[Any] = None
    trace: bool = False
    # The expert's measured joint angles for this episode, when available.
    reference_qpos: Optional[Any] = None

    def __post_init__(self) -> None:
        # Deliberately the drive targets rather than the measured angles: this is
        # what the recorded baseline started from, and the two differ by the
        # controller's lag.
        self.previous = np.asarray(
            self.task.robot.get_left_arm_jointState()[:6], dtype=np.float64
        )
        self._progress = _Progress(self.task, self.reference_qpos)
        self.last_gripper: Optional[float] = None
        self.rows = 0
        self.physics_steps = 0
        self.clamped = 0
        self.max_lag = 0.0
        self.release_row: Optional[int] = None
        self._release_configured = False
        self._gripper_was_closed = False
        self.trace_rows: list[dict] = []

    def step(self, target: np.ndarray) -> None:
        """Execute one 50 Hz target for its grid period."""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError(f"target must be 7-D, got {target.shape}")
        if self.last_gripper is None:
            self.last_gripper = float(target[6])

        arm = target[:6].copy()
        span = arm - self.previous
        limit = self.config.speed_limit_radps * SAMPLE_PERIOD_STEPS * PHYSICS_DT
        norm = float(np.abs(span).max())
        if norm > limit:
            arm = self.previous + span * (limit / norm)
            self.clamped += 1
        self.max_lag = max(self.max_lag, float(np.abs(arm - target[:6]).max()))

        velocity = (
            np.zeros(6)
            if self.config.velocity_mode == "zero"
            else self.config.velocity_gain
            * (arm - self.previous)
            / (SAMPLE_PERIOD_STEPS * PHYSICS_DT)
        )

        gripper = float(target[6])
        if gripper < GRIPPER_CLOSED_OPENING:
            self._gripper_was_closed = True
        elif self._gripper_was_closed and not self._release_configured:
            # The oracle tightens the object's damping and contact solving just
            # before it opens the gripper, and those values are computed at setup
            # but never applied by replaying actions.  Without this the cylinder
            # keeps the loose defaults and rattles in the socket indefinitely.
            self.task._configure_release_contact_solver()
            self._release_configured = True
            self.release_row = self.rows
        if self.ratchet is not None:
            contacts = len(
                self.task.get_gripper_actor_contact_position("panthera_cylinder")
            )
            gripper = self.ratchet(gripper, contacts, self.rows)

        budget = (
            self.config.converge_max_steps
            if self.config.converge_tolerance_rad > 0.0
            else SAMPLE_PERIOD_STEPS
        )
        held = 0
        while held < budget:
            if self.config.interpolate:
                alpha = min((held + 1) / SAMPLE_PERIOD_STEPS, 1.0)
                step_arm = self.previous + (arm - self.previous) * alpha
                step_gripper = self.last_gripper + (gripper - self.last_gripper) * alpha
            else:
                step_arm, step_gripper = arm, gripper
            self.task.robot.set_arm_joints(step_arm, velocity, "left")
            self.task.robot.set_gripper(step_gripper, "left")
            self.task._step_scene()
            held += 1
            if self.config.converge_tolerance_rad > 0.0 and held >= SAMPLE_PERIOD_STEPS:
                measured = np.asarray(
                    self.task._actual_robot_state()["arm_qpos"], dtype=np.float64
                )
                if float(np.abs(measured - arm).max()) <= self.config.converge_tolerance_rad:
                    break

        self.physics_steps += held
        self._progress.observe(self.rows)
        self.last_gripper = gripper
        if self.trace:
            self.trace_rows.append(self._observe_row(arm, gripper))
        self.previous = arm
        self.rows += 1

    def _observe_row(self, arm: np.ndarray, gripper: float) -> dict:
        pose = self.task.cylinder.get_pose()
        # ``get_left_arm_jointState`` returns drive targets and
        # ``get_left_gripper_val`` echoes the last command, so neither can be
        # compared against the command.  ``_actual_robot_state`` is the function
        # the recorder itself used, which makes replay and recording comparable.
        measured = self.task._actual_robot_state()
        achieved = np.asarray(measured["arm_qpos"], dtype=np.float64)
        return {
            "row": self.rows,
            "xyz": [float(v) for v in pose.p],
            "quat": [float(v) for v in pose.q],
            "gripper_cmd": float(gripper),
            "gripper_qpos": float(measured["gripper_qpos"]),
            "contacts": len(
                self.task.get_gripper_actor_contact_position("panthera_cylinder")
            ),
            "arm_cmd": [float(v) for v in arm],
            "arm_qpos": [float(v) for v in achieved],
            "track_err_rad": float(np.abs(achieved - arm).max()),
        }

    def succeeded(self) -> bool:
        return bool(self.task.check_success())

    def finish(self) -> dict:
        """Let the object come to rest, as the oracle did before judging."""
        settled_steps = 0
        if self.config.final_settle:
            if not self._release_configured:
                self.task._configure_release_contact_solver()
                self._release_configured = True
            if not self.task.check_success():
                before = int(getattr(self.task, "simulation_step_count", 0))
                self.task._advance_until_inserted_stable()
                settled_steps = int(getattr(self.task, "simulation_step_count", 0)) - before
        result = {
            "success": bool(self.task.check_success()),
            "executed_actions": self.rows,
            "physics_steps": self.physics_steps,
            "clamped": self.clamped,
            "max_speed_limit_lag_rad": self.max_lag,
            "release_solver_row": self.release_row,
            "final_settle_steps": settled_steps,
            # Success is one bit, and a policy that never reaches the object
            # produces the same bit whether it moved a centimetre or a metre.
            # These say how far along the chain it got.
            "progress": self._progress.report(),
            "metrics": {
                key: (float(value) if not isinstance(value, bool) else value)
                for key, value in self.task.success_metrics().items()
            },
        }
        if self.ratchet is not None and hasattr(self.ratchet, "telemetry"):
            result["ratchet"] = self.ratchet.telemetry()
        if self.trace:
            result["trace"] = self.trace_rows
        return result
