"""CPU-only contracts for release reward physical-replay variants."""

from __future__ import annotations

import numpy as np

from audit_release_reward_replay import build_case_actions, gate_results


def test_case_actions_share_arm_suffix_and_only_change_gripper() -> None:
    actions = np.arange(12 * 7, dtype=np.float64).reshape(12, 7) / 100.0
    actions[:, 6] = np.linspace(0.2, 0.9, len(actions))

    cases = build_case_actions(actions, prefix_actions=5)

    expert = cases["expert_suffix"]
    assert expert.shape == (7, 7)
    assert np.array_equal(cases["immediate_open"][:, :6], expert[:, :6])
    assert np.array_equal(cases["never_open"][:, :6], expert[:, :6])
    assert np.all(cases["immediate_open"][:, 6] == 1.0)
    assert np.all(cases["never_open"][:, 6] == actions[4, 6])
    assert np.array_equal(actions[:, 6], np.linspace(0.2, 0.9, len(actions)))


def test_gate_requires_expert_success_and_strict_r1_r2_ranking() -> None:
    passing = {
        "expert_suffix": {
            "case": "expert_suffix",
            "success": True,
            "r1_return": 1.0,
            "r2_return": 1.1,
        },
        "immediate_open": {
            "case": "immediate_open",
            "success": False,
            "r1_return": -1.0,
            "r2_return": -1.1,
        },
        "never_open": {
            "case": "never_open",
            "success": False,
            "r1_return": 0.0,
            "r2_return": -0.01,
        },
    }
    assert gate_results(passing) == (True, [])

    passing["never_open"]["r2_return"] = 1.2
    passed, reasons = gate_results(passing)
    assert passed is False
    assert any("r2_return does not rank" in reason for reason in reasons)
