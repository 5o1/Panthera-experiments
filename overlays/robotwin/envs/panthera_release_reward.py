"""Task-effect rewards for Panthera cylinder insertion RL.

The policy never observes this module's privileged fields.  They are derived
from simulator geometry/contact state and are used only for reward and
evaluation.  In particular, the evaluator does not reward a gripper command,
an expert-frame index, or a hand-authored action phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence


RewardVariant = Literal["r1", "r2"]


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class ReleaseEffectState:
    """Privileged physical effects at one policy action boundary.

    ``grasped`` means the cylinder is reliably following the end effector, not
    merely that one contact point exists.  ``released_in_valid_volume`` means
    contact has ended inside the calibrated release volume.  ``success`` must
    already include the environment's consecutive settling requirement.
    """

    tcp_to_object_m: float
    target_xy_error_m: float
    target_height_error_m: float
    target_axis_error_deg: float
    object_linear_speed_mps: float
    object_angular_speed_radps: float
    grasped: bool
    released_in_valid_volume: bool
    success: bool
    hard_failure: bool

    def __post_init__(self) -> None:
        for name in (
            "tcp_to_object_m",
            "target_xy_error_m",
            "target_height_error_m",
            "target_axis_error_deg",
            "object_linear_speed_mps",
            "object_angular_speed_radps",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.success and self.hard_failure:
            raise ValueError("a state cannot be both successful and a hard failure")
        if self.success and not self.released_in_valid_volume:
            raise ValueError("success requires a valid physical release")
        if self.grasped and self.released_in_valid_volume:
            raise ValueError("a released cylinder cannot remain reliably grasped")


@dataclass(frozen=True)
class ReleaseRewardConfig:
    """Bounded reward scales shared by R1 and potential-shaped R2."""

    gamma: float = 0.99
    potential_scale: float = 0.25
    shaping_coefficient: float = 1.0
    time_penalty: float = 2.0e-4
    arm_action_change_coefficient: float = 2.0e-3
    maximum_arm_action_change_penalty: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        for name in (
            "potential_scale",
            "shaping_coefficient",
            "time_penalty",
            "arm_action_change_coefficient",
            "maximum_arm_action_change_penalty",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.potential_scale > 0.25:
            raise ValueError("potential_scale must not exceed one quarter of success")


@dataclass(frozen=True)
class ReleaseReward:
    """One auditable reward decomposition."""

    total: float
    task: float
    shaping: float
    time: float
    arm_action_change: float
    previous_potential: float
    next_potential: float


def _unit_ratio(value: float, scale: float) -> float:
    return min(_finite_nonnegative("metric", value) / scale, 1.0)


def remaining_cost(state: ReleaseEffectState) -> float:
    """Estimate normalized remaining task cost from physical effects only.

    Stage offsets are monotone: a reliable grasp is cheaper than approach, a
    valid release is cheaper than transport, and settled insertion is zero.
    Invalid release/loss maps to maximum cost, so changing contact outside the
    target volume cannot manufacture progress reward.
    """

    if state.success:
        return 0.0
    if state.hard_failure:
        return 1.0

    xy = _unit_ratio(state.target_xy_error_m, 0.20)
    height = _unit_ratio(state.target_height_error_m, 0.20)
    axis = _unit_ratio(state.target_axis_error_deg, 90.0)
    motion = 0.5 * (
        _unit_ratio(state.object_linear_speed_mps, 0.25)
        + _unit_ratio(state.object_angular_speed_radps, 2.0)
    )

    if state.released_in_valid_volume:
        # After a valid release, only final pose and settling remain.
        return min(0.10 + 0.15 * xy + 0.15 * height + 0.10 * axis + 0.10 * motion, 1.0)
    if state.grasped:
        # Transport/alignment is downstream of grasping but still retains a
        # nonzero release-and-settle cost.
        return min(0.30 + 0.20 * xy + 0.15 * height + 0.15 * axis + 0.05 * motion, 1.0)

    approach = _unit_ratio(state.tcp_to_object_m, 0.25)
    return min(0.65 + 0.25 * approach + 0.05 * xy + 0.05 * axis, 1.0)


def potential(
    state: ReleaseEffectState,
    config: ReleaseRewardConfig = ReleaseRewardConfig(),
) -> float:
    """Return ``Phi(s)`` in ``[-potential_scale, 0]``."""

    return -config.potential_scale * remaining_cost(state)


def _arm_change(
    previous_action: Sequence[float] | None,
    current_action: Sequence[float] | None,
) -> float:
    if previous_action is None or current_action is None:
        return 0.0
    previous = tuple(float(value) for value in previous_action)
    current = tuple(float(value) for value in current_action)
    if len(previous) < 6 or len(current) < 6:
        raise ValueError("actions must contain at least six arm joints")
    arm_delta = [abs(current[index] - previous[index]) for index in range(6)]
    if not all(math.isfinite(value) for value in arm_delta):
        raise ValueError("arm action changes must be finite")
    return sum(arm_delta) / 6.0


def evaluate_transition(
    previous: ReleaseEffectState,
    current: ReleaseEffectState,
    *,
    variant: RewardVariant,
    previous_action: Sequence[float] | None = None,
    current_action: Sequence[float] | None = None,
    config: ReleaseRewardConfig = ReleaseRewardConfig(),
) -> ReleaseReward:
    """Evaluate one physical transition for sparse R1 or shaped R2.

    Success and hard-failure rewards are edge-triggered so absorbing states do
    not pay repeatedly.  The smoothness term uses only the six arm joints;
    excluding the gripper preserves a legitimate release transition.
    """

    if variant not in {"r1", "r2"}:
        raise ValueError(f"unsupported reward variant: {variant!r}")
    task = 0.0
    if current.success and not previous.success:
        task = 1.0
    elif current.hard_failure and not previous.hard_failure:
        task = -1.0

    previous_phi = potential(previous, config)
    current_phi = potential(current, config)
    if variant == "r1":
        shaping = 0.0
        time_cost = 0.0
        change_cost = 0.0
    else:
        shaping = config.shaping_coefficient * (
            config.gamma * current_phi - previous_phi
        )
        time_cost = 0.0 if (current.success or current.hard_failure) else config.time_penalty
        change_cost = min(
            config.arm_action_change_coefficient
            * _arm_change(previous_action, current_action),
            config.maximum_arm_action_change_penalty,
        )

    total = task + shaping - time_cost - change_cost
    return ReleaseReward(
        total=float(total),
        task=float(task),
        shaping=float(shaping),
        time=float(-time_cost),
        arm_action_change=float(-change_cost),
        previous_potential=float(previous_phi),
        next_potential=float(current_phi),
    )
