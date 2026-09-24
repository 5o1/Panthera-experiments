import numpy as np

from envs.place_cylinder_in_groove import place_cylinder_in_groove
from envs.panthera_release_reward import ReleaseEffectState


class _Robot:
    def __init__(self):
        self.arm_calls = []
        self.gripper_calls = []

    def set_arm_joints(self, position, velocity, arm):
        self.arm_calls.append((np.asarray(position), np.asarray(velocity), arm))

    def set_gripper(self, opening, arm):
        self.gripper_calls.append((float(opening), arm))


def _task(*, step_limit=8, success_after_physics_steps=None):
    task = object.__new__(place_cylinder_in_groove)
    task.robot = _Robot()
    task.step_lim = step_limit
    task.take_action_cnt = 0
    task.eval_success = False
    task.physics_steps = 0
    task.release_configurations = 0
    task._policy_release_solver_configured = False
    task._rl_previous_action = None
    task._reset_policy_episode_metrics()
    task._actual_robot_state = lambda: {
        "arm_qpos": np.zeros(6),
        "gripper_qpos": 0.2,
    }
    task._step_scene = lambda: setattr(
        task, "physics_steps", task.physics_steps + 1
    )
    task.check_success = lambda: (
        success_after_physics_steps is not None
        and task.physics_steps >= success_after_physics_steps
    )
    task._configure_release_contact_solver = lambda: setattr(
        task, "release_configurations", task.release_configurations + 1
    )
    return task


def _effect(**changes):
    values = {
        "tcp_to_object_m": 0.20,
        "target_xy_error_m": 0.18,
        "target_height_error_m": 0.15,
        "target_axis_error_deg": 45.0,
        "object_linear_speed_mps": 0.0,
        "object_angular_speed_radps": 0.0,
        "grasped": False,
        "released_in_valid_volume": False,
        "success": False,
        "hard_failure": False,
    }
    values.update(changes)
    return ReleaseEffectState(**values)


def test_sparse_chunk_uses_five_physics_ticks_per_50hz_action():
    task = _task()
    actions = np.zeros((2, 7), dtype=np.float64)
    actions[0, 6] = 0.2
    actions[1, 6] = 0.8

    reward, termination, truncation, info = task.gen_sparse_reward_data(actions)

    assert reward.tolist() == [0.0]
    assert termination.tolist() == [0]
    assert truncation.tolist() == [0]
    assert info == {"success": False, "executed_steps": 2}
    assert task.physics_steps == 10
    assert len(task.robot.arm_calls) == 10
    assert len(task.robot.gripper_calls) == 10
    assert task.take_action_cnt == 2
    assert task.release_configurations == 1


def test_sparse_chunk_stops_at_success_and_reports_consumed_actions():
    task = _task(success_after_physics_steps=5)
    actions = np.zeros((3, 7), dtype=np.float64)

    reward, termination, truncation, info = task.gen_sparse_reward_data(actions)

    assert reward.tolist() == [1.0]
    assert termination.tolist() == [1]
    assert truncation.tolist() == [0]
    assert info == {"success": True, "executed_steps": 1}
    assert task.physics_steps == 5
    assert task.take_action_cnt == 1


def test_sparse_chunk_truncates_at_action_budget():
    task = _task(step_limit=1)

    _, termination, truncation, info = task.gen_sparse_reward_data(
        np.zeros((3, 7), dtype=np.float64)
    )

    assert termination.tolist() == [0]
    assert truncation.tolist() == [1]
    assert info["executed_steps"] == 1
    assert task.physics_steps == 5


def test_effect_reward_preserves_action_aligned_r2_values():
    task = _task(step_limit=8)
    task.rl_reward_variant = "r2"
    effects = iter(
        [
            _effect(),
            _effect(tcp_to_object_m=0.01, grasped=True),
            _effect(
                tcp_to_object_m=0.08,
                target_xy_error_m=0.002,
                target_height_error_m=0.002,
                target_axis_error_deg=1.0,
                released_in_valid_volume=True,
                success=True,
            ),
        ]
    )
    task.policy_release_effect_state = lambda **_: next(effects)

    reward, termination, truncation, info = task.gen_sparse_reward_data(
        np.zeros((3, 7), dtype=np.float64)
    )

    assert termination.tolist() == [1]
    assert truncation.tolist() == [0]
    assert info["success"] is True
    assert info["executed_steps"] == 2
    assert len(info["step_rewards"]) == 2
    assert len(info["reward_components"]) == 2
    assert info["panthera_grasped_once"] == 1.0
    assert info["panthera_valid_release_once"] == 1.0
    assert info["panthera_invalid_release_once"] == 0.0
    assert info["panthera_release_delay_censored"] == 0.0
    assert reward[0] == np.float32(sum(info["step_rewards"]))


def test_effect_metrics_separate_invalid_release_and_chunk_boundary_jitter():
    task = _task(step_limit=8)
    task.rl_reward_variant = "r2"
    effects = iter(
        [
            _effect(
                tcp_to_object_m=0.01,
                target_xy_error_m=0.005,
                target_height_error_m=0.010,
                target_axis_error_deg=2.0,
                grasped=True,
            ),
            _effect(
                tcp_to_object_m=0.01,
                target_xy_error_m=0.005,
                target_height_error_m=0.010,
                target_axis_error_deg=2.0,
                grasped=True,
            ),
            _effect(
                tcp_to_object_m=0.01,
                target_xy_error_m=0.005,
                target_height_error_m=0.010,
                target_axis_error_deg=2.0,
                grasped=True,
            ),
            _effect(
                tcp_to_object_m=0.01,
                target_xy_error_m=0.005,
                target_height_error_m=0.010,
                target_axis_error_deg=2.0,
                grasped=True,
            ),
            _effect(
                tcp_to_object_m=0.10,
                target_xy_error_m=0.080,
                target_height_error_m=0.030,
                target_axis_error_deg=20.0,
                grasped=False,
            ),
        ]
    )
    task.policy_release_effect_state = lambda **_: next(effects)

    first = np.zeros((2, 7), dtype=np.float64)
    first[0, 0] = 0.01
    first[1, 0] = 0.02
    task.gen_sparse_reward_data(first)
    second = np.zeros((1, 7), dtype=np.float64)
    second[0, 0] = 0.20
    _, _, _, info = task.gen_sparse_reward_data(second)

    assert info["panthera_target_grasped_once"] == 1.0
    assert info["panthera_valid_release_once"] == 0.0
    assert info["panthera_invalid_release_once"] == 1.0
    assert info["panthera_release_delay_actions"] == 3.0
    assert info["panthera_release_delay_censored"] == 1.0
    assert info["panthera_chunk_boundary_second_difference_l1_mean"] > 0.0
    assert info["panthera_chunk_boundary_second_difference_l1_max"] > 0.0
