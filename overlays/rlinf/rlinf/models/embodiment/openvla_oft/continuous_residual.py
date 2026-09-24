"""Continuous residual policy utilities for OpenVLA-OFT PPO.

The supervised L1 head is deterministic, so it cannot provide the action
log-probabilities required by PPO.  This module keeps that head as a frozen
prior and learns a small, tanh-squashed Gaussian residual policy around it.
All calculations stay in OpenVLA-OFT's normalized action space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


_ATANH_EPS = 1.0e-6


def _atanh(value: torch.Tensor) -> torch.Tensor:
    """Numerically stable float32 inverse tanh for normalized actions.

    ``1 - 1e-6`` rounds back to exactly ``1`` in bfloat16.  Clamping before
    promotion therefore still feeds ``atanh(1)`` at saturated SFT actions,
    which first appears near the gripper-release boundary and poisons PPO with
    infinite log-probabilities.  Promote before applying the open-interval
    clamp so the epsilon is representable.
    """
    value = value.float().clamp(-1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def squashed_gaussian_log_prob(
    action: torch.Tensor,
    location: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    """Return stable float32 log-probability after a tanh transform.

    OpenVLA runs its forward pass in bf16, while PPO ratios exponentiate the
    difference between current and rollout log-probabilities.  Performing the
    density and tanh-Jacobian arithmetic in bf16 is both numerically fragile
    and rejected by RLinf's loss contract.  ``Tensor.float()`` remains
    differentiable, so this promotion does not disconnect the residual head.
    """
    action_fp32 = action.float()
    location_fp32 = location.float()
    log_std_fp32 = log_std.float()
    pre_tanh = _atanh(action_fp32)
    inverse_std = torch.exp(-log_std_fp32)
    normal_log_prob = (
        -0.5 * ((pre_tanh - location_fp32) * inverse_std).square()
        - log_std_fp32
        - 0.5 * math.log(2.0 * math.pi)
    )
    correction = torch.log(1.0 - action_fp32.square() + _ATANH_EPS)
    return normal_log_prob - correction


@dataclass(frozen=True)
class ResidualPolicyOutput:
    """Distribution parameters and its deterministic squashed mean."""

    location: torch.Tensor
    log_std: torch.Tensor
    mean_action: torch.Tensor


class SquashedGaussianResidualPolicy(nn.Module):
    """A bounded stochastic residual around a frozen OpenVLA-OFT L1 prior.

    ``variant`` controls which temporal information is used:

    * ``c0``: no chunk-history input and no overlap regularizer;
    * ``c1``: no history input; the caller may add overlap consistency loss;
    * ``c2``: condition the residual on the aligned previous chunk and the
      most recently executed normalized action.
    """

    VALID_VARIANTS = frozenset({"c0", "c1", "c2"})

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        num_action_chunks: int,
        variant: str,
        feature_dim: int = 256,
        history_dim: int = 64,
        max_residual: float = 0.25,
        initial_log_std: float = -3.0,
        min_log_std: float = -5.0,
        max_log_std: float = -1.0,
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"continuous RL variant must be one of {sorted(self.VALID_VARIANTS)}, "
                f"got {variant!r}"
            )
        if not 0.0 < max_residual <= 1.0:
            raise ValueError("max_residual must be in (0, 1]")
        if not min_log_std <= initial_log_std <= max_log_std:
            raise ValueError("initial_log_std must lie inside the configured bounds")

        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.variant = variant
        self.max_residual = float(max_residual)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.observation_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )
        policy_input_dim = feature_dim
        if variant == "c2":
            self.history_encoder = nn.Sequential(
                nn.Linear(2 * action_dim + 1, history_dim),
                nn.GELU(),
            )
            policy_input_dim += history_dim
        else:
            self.history_encoder = None

        self.residual_head = nn.Linear(policy_input_dim, action_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(initial_log_std), dtype=torch.float32)
        )

    def _reshape_action_features(
        self, action_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, token_count, hidden_dim = action_hidden_states.shape
        expected_tokens = self.num_action_chunks * self.action_dim
        if token_count != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} action tokens, got {token_count}"
            )
        return action_hidden_states.reshape(
            batch_size, self.num_action_chunks, self.action_dim, hidden_dim
        ).mean(dim=2)

    def _history_features(
        self,
        previous_mean: torch.Tensor,
        previous_executed_action: torch.Tensor,
        previous_valid: torch.Tensor,
        execution_horizon: int,
        *,
        perturb: bool,
        noise_std: float,
        dropout_probability: float,
    ) -> torch.Tensor:
        if not 1 <= execution_horizon <= self.num_action_chunks:
            raise ValueError("execution_horizon is outside the action chunk")
        batch_size = previous_mean.shape[0]
        aligned = previous_mean.new_zeros(
            batch_size, self.num_action_chunks, self.action_dim
        )
        overlap = self.num_action_chunks - execution_horizon
        if overlap > 0:
            aligned[:, :overlap] = previous_mean[:, execution_horizon:]

        validity = previous_valid.to(dtype=previous_mean.dtype).reshape(batch_size, 1, 1)
        aligned = aligned * validity
        last_action = previous_executed_action.reshape(
            batch_size, 1, self.action_dim
        ).expand(-1, self.num_action_chunks, -1)
        last_action = last_action * validity

        if perturb and noise_std > 0.0:
            aligned = aligned + torch.randn_like(aligned) * noise_std * validity
            last_action = last_action + torch.randn_like(last_action) * noise_std * validity
        if perturb and dropout_probability > 0.0:
            keep = (
                torch.rand(batch_size, 1, 1, device=previous_mean.device)
                >= dropout_probability
            ).to(previous_mean.dtype)
            aligned = aligned * keep
            last_action = last_action * keep
            validity = validity * keep

        return torch.cat(
            (
                aligned,
                last_action,
                validity.expand(-1, self.num_action_chunks, -1),
            ),
            dim=-1,
        )

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        sft_mean: torch.Tensor,
        *,
        previous_mean: torch.Tensor,
        previous_executed_action: torch.Tensor,
        previous_valid: torch.Tensor,
        execution_horizon: int,
        perturb_history: bool = False,
        history_noise_std: float = 0.0,
        history_dropout_probability: float = 0.0,
    ) -> ResidualPolicyOutput:
        expected_shape = (
            action_hidden_states.shape[0],
            self.num_action_chunks,
            self.action_dim,
        )
        if tuple(sft_mean.shape) != expected_shape:
            raise ValueError(
                f"SFT mean has shape {tuple(sft_mean.shape)}, expected {expected_shape}"
            )

        features = self.observation_encoder(
            self._reshape_action_features(action_hidden_states)
        )
        if self.history_encoder is not None:
            history = self._history_features(
                previous_mean,
                previous_executed_action,
                previous_valid,
                execution_horizon,
                perturb=perturb_history,
                noise_std=history_noise_std,
                dropout_probability=history_dropout_probability,
            )
            # Rollout history is intentionally stored in float32 for stable
            # action bookkeeping, while OpenVLA's residual module normally
            # runs in bf16.  Linear layers require matching input/weight
            # dtypes, so align the derived history features with the encoded
            # observation immediately before the trainable history encoder.
            history = history.to(device=features.device, dtype=features.dtype)
            features = torch.cat((features, self.history_encoder(history)), dim=-1)

        residual = self.max_residual * torch.tanh(self.residual_head(features))
        location = _atanh(sft_mean) + residual
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std).to(
            dtype=location.dtype, device=location.device
        )
        log_std = log_std.reshape(1, 1, self.action_dim).expand_as(location)
        return ResidualPolicyOutput(
            location=location,
            log_std=log_std,
            mean_action=torch.tanh(location),
        )

    def sample(
        self, output: ResidualPolicyOutput, *, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            pre_tanh = output.location
        else:
            pre_tanh = output.location + torch.exp(output.log_std) * torch.randn_like(
                output.location
            )
        action = torch.tanh(pre_tanh)
        log_prob = squashed_gaussian_log_prob(
            action, output.location, output.log_std
        )
        return action, log_prob

    def log_prob(
        self, action: torch.Tensor, output: ResidualPolicyOutput
    ) -> torch.Tensor:
        if action.ndim != output.location.ndim:
            raise ValueError(
                "action and residual distribution must have the same rank, "
                f"got {action.ndim} and {output.location.ndim}"
            )
        if (
            action.shape[0] != output.location.shape[0]
            or action.shape[2:] != output.location.shape[2:]
        ):
            raise ValueError(
                "action and residual distribution disagree outside the time "
                f"dimension: {tuple(action.shape)} versus "
                f"{tuple(output.location.shape)}"
            )
        horizon = action.shape[1]
        if not 1 <= horizon <= output.location.shape[1]:
            raise ValueError(
                f"action horizon {horizon} is outside distribution horizon "
                f"{output.location.shape[1]}"
            )
        return squashed_gaussian_log_prob(
            action,
            output.location[:, :horizon],
            output.log_std[:, :horizon],
        )


def overlap_consistency_loss(
    current_mean: torch.Tensor,
    previous_mean: torch.Tensor,
    previous_valid: torch.Tensor,
    *,
    execution_horizon: int,
    arm_dim: int = 6,
) -> torch.Tensor:
    """MSE between time-aligned overlapping arm predictions.

    The gripper channel is deliberately excluded: opening versus closing is a
    semantic transition, so smoothing it can suppress the release action.
    """
    chunk_length = current_mean.shape[1]
    overlap = chunk_length - int(execution_horizon)
    if overlap <= 0:
        return current_mean.sum() * 0.0
    valid = previous_valid.to(dtype=current_mean.dtype).reshape(-1, 1, 1)
    squared_error = (
        current_mean[:, :overlap, :arm_dim]
        - previous_mean[:, execution_horizon:, :arm_dim]
    ).square()
    denominator = valid.sum() * overlap * arm_dim
    if float(denominator.detach().cpu()) == 0.0:
        return current_mean.sum() * 0.0
    return (squared_error * valid).sum() / denominator
