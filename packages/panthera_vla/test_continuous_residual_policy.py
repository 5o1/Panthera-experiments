"""CPU regression tests for the OpenVLA-OFT continuous residual PPO head."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "overlays/rlinf/rlinf/models/embodiment/openvla_oft/continuous_residual.py"
)
SPEC = importlib.util.spec_from_file_location("continuous_residual", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inputs(batch_size: int = 3):
    hidden = torch.randn(batch_size, 25 * 7, 32)
    sft_mean = torch.tanh(torch.randn(batch_size, 25, 7) * 0.2)
    previous_mean = torch.tanh(torch.randn(batch_size, 25, 7) * 0.2)
    previous_action = previous_mean[:, 4]
    previous_valid = torch.tensor([True, False, True])[:batch_size]
    return hidden, sft_mean, previous_mean, previous_action, previous_valid


@pytest.mark.parametrize("variant", ["c0", "c1", "c2"])
def test_policy_initializes_at_sft_mean_and_has_finite_log_prob(variant: str):
    policy = MODULE.SquashedGaussianResidualPolicy(
        hidden_dim=32,
        action_dim=7,
        num_action_chunks=25,
        variant=variant,
        feature_dim=16,
        history_dim=8,
    )
    hidden, sft_mean, previous_mean, previous_action, previous_valid = _inputs()
    output = policy(
        hidden,
        sft_mean,
        previous_mean=previous_mean,
        previous_executed_action=previous_action,
        previous_valid=previous_valid,
        execution_horizon=5,
    )

    assert torch.allclose(output.mean_action, sft_mean, atol=2.0e-6)
    deterministic_actions, deterministic_log_prob = policy.sample(
        output, deterministic=True
    )
    assert torch.allclose(deterministic_actions, sft_mean, atol=2.0e-6)
    assert torch.isfinite(deterministic_log_prob).all()
    actions, log_prob = policy.sample(output, deterministic=False)
    assert actions.shape == (3, 25, 7)
    assert log_prob.shape == actions.shape
    assert log_prob.dtype == torch.float32
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()


def test_recomputed_log_prob_matches_rollout_log_prob():
    policy = MODULE.SquashedGaussianResidualPolicy(
        hidden_dim=32,
        action_dim=7,
        num_action_chunks=25,
        variant="c2",
        feature_dim=16,
        history_dim=8,
    )
    inputs = _inputs()
    output = policy(
        inputs[0],
        inputs[1],
        previous_mean=inputs[2],
        previous_executed_action=inputs[3],
        previous_valid=inputs[4],
        execution_horizon=5,
    )
    actions, rollout_log_prob = policy.sample(output, deterministic=False)
    recomputed = policy.log_prob(actions, output)
    assert torch.allclose(recomputed, rollout_log_prob, atol=1.0e-6)


def test_recomputed_log_prob_accepts_only_the_executed_horizon():
    policy = MODULE.SquashedGaussianResidualPolicy(
        hidden_dim=32,
        action_dim=7,
        num_action_chunks=25,
        variant="c0",
        feature_dim=16,
        history_dim=8,
    )
    inputs = _inputs()
    output = policy(
        inputs[0],
        inputs[1],
        previous_mean=inputs[2],
        previous_executed_action=inputs[3],
        previous_valid=inputs[4],
        execution_horizon=5,
    )
    actions, rollout_log_prob = policy.sample(output, deterministic=False)

    recomputed = policy.log_prob(actions[:, :5], output)

    assert recomputed.shape == (3, 5, 7)
    assert recomputed.dtype == torch.float32
    assert torch.allclose(recomputed, rollout_log_prob[:, :5], atol=1.0e-6)


def test_bfloat16_policy_log_prob_is_promoted_and_keeps_gradients():
    action = torch.tanh(torch.randn(2, 5, 7, dtype=torch.bfloat16))
    location = torch.randn(2, 5, 7, dtype=torch.bfloat16, requires_grad=True)
    log_std = torch.full(
        (2, 5, 7), -3.0, dtype=torch.bfloat16, requires_grad=True
    )

    log_prob = MODULE.squashed_gaussian_log_prob(action, location, log_std)
    log_prob.mean().backward()

    assert log_prob.dtype == torch.float32
    assert location.grad is not None
    assert log_std.grad is not None
    assert torch.isfinite(location.grad).all()
    assert torch.isfinite(log_std.grad).all()


def test_bfloat16_sft_actions_at_exact_bounds_stay_finite_and_differentiable():
    policy = MODULE.SquashedGaussianResidualPolicy(
        hidden_dim=32,
        action_dim=7,
        num_action_chunks=25,
        variant="c0",
        feature_dim=16,
        history_dim=8,
    ).to(dtype=torch.bfloat16)
    hidden = torch.randn(2, 25 * 7, 32, dtype=torch.bfloat16)
    sft_mean = torch.ones(2, 25, 7, dtype=torch.bfloat16)
    sft_mean[1].mul_(-1.0)
    previous_mean = torch.zeros_like(sft_mean)
    previous_action = torch.zeros(2, 7, dtype=torch.bfloat16)
    previous_valid = torch.zeros(2, dtype=torch.bool)

    output = policy(
        hidden,
        sft_mean,
        previous_mean=previous_mean,
        previous_executed_action=previous_action,
        previous_valid=previous_valid,
        execution_horizon=5,
    )
    actions, log_prob = policy.sample(output, deterministic=False)
    (-log_prob.mean()).backward()

    assert output.location.dtype == torch.float32
    assert torch.isfinite(output.location).all()
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()
    assert policy.residual_head.weight.grad is not None
    assert torch.isfinite(policy.residual_head.weight.grad).all()


def test_bfloat16_c2_accepts_float32_rollout_history():
    """C2 must accept the float32 history retained by the rollout worker."""
    policy = MODULE.SquashedGaussianResidualPolicy(
        hidden_dim=32,
        action_dim=7,
        num_action_chunks=25,
        variant="c2",
        feature_dim=16,
        history_dim=8,
    ).to(dtype=torch.bfloat16)
    hidden = torch.randn(2, 25 * 7, 32, dtype=torch.bfloat16)
    sft_mean = torch.tanh(torch.randn(2, 25, 7, dtype=torch.bfloat16))
    previous_mean = torch.tanh(torch.randn(2, 25, 7, dtype=torch.float32))
    previous_action = previous_mean[:, 4]
    previous_valid = torch.tensor([True, False])

    output = policy(
        hidden,
        sft_mean,
        previous_mean=previous_mean,
        previous_executed_action=previous_action,
        previous_valid=previous_valid,
        execution_horizon=5,
        perturb_history=True,
        history_noise_std=0.01,
        history_dropout_probability=0.1,
    )
    actions, log_prob = policy.sample(output, deterministic=False)
    (-log_prob.mean()).backward()

    assert output.location.dtype == torch.float32
    assert actions.dtype == torch.float32
    assert log_prob.dtype == torch.float32
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()
    assert policy.history_encoder[0].weight.grad is not None
    assert torch.isfinite(policy.history_encoder[0].weight.grad).all()


def test_consistency_aligns_previous_tail_with_current_prefix_and_ignores_gripper():
    previous = torch.zeros(2, 25, 7)
    previous[:, :, :6] = torch.arange(25).reshape(1, 25, 1)
    current = torch.zeros_like(previous)
    current[:, :20, :6] = previous[:, 5:, :6]
    current[:, :, 6] = 1.0 - previous[:, :, 6]
    valid = torch.tensor([True, False])

    loss = MODULE.overlap_consistency_loss(
        current,
        previous,
        valid,
        execution_horizon=5,
        arm_dim=6,
    )
    assert loss.item() == pytest.approx(0.0)

    current[0, 0, 0] += 1.0
    changed = MODULE.overlap_consistency_loss(
        current,
        previous,
        valid,
        execution_horizon=5,
        arm_dim=6,
    )
    assert changed.item() > 0.0


def test_history_perturbation_does_not_mutate_inputs():
    policy = MODULE.SquashedGaussianResidualPolicy(
        hidden_dim=32,
        action_dim=7,
        num_action_chunks=25,
        variant="c2",
        feature_dim=16,
        history_dim=8,
    )
    hidden, sft_mean, previous_mean, previous_action, previous_valid = _inputs()
    before_mean = previous_mean.clone()
    before_action = previous_action.clone()
    policy(
        hidden,
        sft_mean,
        previous_mean=previous_mean,
        previous_executed_action=previous_action,
        previous_valid=previous_valid,
        execution_horizon=5,
        perturb_history=True,
        history_noise_std=0.05,
        history_dropout_probability=0.5,
    )
    assert torch.equal(previous_mean, before_mean)
    assert torch.equal(previous_action, before_action)
