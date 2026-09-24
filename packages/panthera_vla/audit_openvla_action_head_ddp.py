#!/usr/bin/env python3
"""Verify that OpenVLA-OFT action-head gradients stay synchronized under DDP.

Run this file with ``torchrun`` and at least two CUDA workers.  Each rank sees
a deliberately different local batch.  The negative control calls the method
on ``DDP.module`` and must diverge; the production path calls the DDP wrapper
and must keep gradients, parameters, and predictions identical across ranks.
"""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Callable

os.environ.setdefault("ROBOT_PLATFORM", "PANTHERA")
os.environ.setdefault("PANTHERA_ACTION_CHUNK", "25")
os.environ.setdefault("PANTHERA_PROPRIO_DIM", "28")

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead


SYNC_ATOL = 1.0e-7
DIVERGENCE_ATOL = 1.0e-7


def _rank_max_abs_difference(value: torch.Tensor) -> float:
    """Return the largest absolute difference between rank 0 and any rank."""
    copies = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(copies, value)
    reference = copies[0]
    return max(float((reference - other).abs().max()) for other in copies[1:])


def _selected_weight(head: nn.Module, kind: str) -> torch.Tensor:
    if kind == "l1":
        return head.model.fc2.weight
    return head.noise_predictor.mlp_resnet.fc2.weight


def _predict_directly(head: nn.Module, kind: str, hidden: torch.Tensor) -> torch.Tensor:
    if kind == "l1":
        return head.predict_action(hidden)
    return head.predict_noise(hidden)


def _new_head(kind: str, device: torch.device) -> nn.Module:
    if kind == "l1":
        head = L1RegressionActionHead(input_dim=64, hidden_dim=128, action_dim=7)
    else:
        head = DiffusionActionHead(
            input_dim=64,
            hidden_dim=64,
            action_dim=7,
            num_diffusion_steps_train=10,
        )
    return head.to(device)


def _run_case(*, kind: str, bypass_ddp: bool, device: torch.device) -> dict[str, float]:
    rank = dist.get_rank()
    torch.manual_seed(20260922)
    torch.cuda.manual_seed_all(20260922)

    wrapped = DDP(
        _new_head(kind, device),
        device_ids=[device.index],
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=5.0e-4)

    # Different rank-local batches are what make missing all-reduce observable.
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    hidden = torch.randn(6, 25 * 7, 64, generator=generator, device=device)
    target = torch.randn(6, 25, 7, generator=generator, device=device)

    optimizer.zero_grad(set_to_none=True)
    prediction = (
        _predict_directly(wrapped.module, kind, hidden)
        if bypass_ddp
        else wrapped(hidden)
    )
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    loss_function = nn.functional.l1_loss if kind == "l1" else nn.functional.mse_loss
    loss = loss_function(prediction, target)
    loss.backward()

    weight = _selected_weight(wrapped.module, kind)
    gradient_difference = _rank_max_abs_difference(weight.grad.detach())
    optimizer.step()
    parameter_difference = _rank_max_abs_difference(weight.detach())

    common_generator = torch.Generator(device=device).manual_seed(424242)
    common_hidden = torch.randn(1, 25 * 7, 64, generator=common_generator, device=device)
    with torch.no_grad():
        common_output = _predict_directly(wrapped.module, kind, common_hidden)
    output_difference = _rank_max_abs_difference(common_output)

    losses = [torch.empty_like(loss.detach()) for _ in range(dist.get_world_size())]
    dist.all_gather(losses, loss.detach())
    return {
        "local_loss_min": min(float(item) for item in losses),
        "local_loss_max": max(float(item) for item in losses),
        "gradient_rank_max_abs_diff": gradient_difference,
        "parameter_rank_max_abs_diff": parameter_difference,
        "common_output_rank_max_abs_diff": output_difference,
    }


def _assert_result(kind: str, result: dict[str, dict[str, float]]) -> None:
    for metric in (
        "gradient_rank_max_abs_diff",
        "parameter_rank_max_abs_diff",
        "common_output_rank_max_abs_diff",
    ):
        negative_value = result["module_bypass_negative_control"][metric]
        fixed_value = result["ddp_forward"][metric]
        if negative_value <= DIVERGENCE_ATOL:
            raise AssertionError(
                f"{kind} negative control did not diverge for {metric}: {negative_value}"
            )
        if fixed_value > SYNC_ATOL:
            raise AssertionError(
                f"{kind} DDP path is not synchronized for {metric}: {fixed_value}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write rank-0 JSON evidence here")
    args = parser.parse_args()

    # Importing this OpenVLA-OFT build may initialize Accelerate's process
    # group. Reuse it instead of attempting to initialize the group twice.
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() < 2:
        raise RuntimeError("DDP synchronization audit requires at least two ranks")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cases: dict[str, dict[str, dict[str, float]]] = {}
    for kind in ("l1", "diffusion"):
        result = {
            "module_bypass_negative_control": _run_case(
                kind=kind, bypass_ddp=True, device=device
            ),
            "ddp_forward": _run_case(kind=kind, bypass_ddp=False, device=device),
        }
        _assert_result(kind, result)
        cases[kind] = result

    payload = {
        "schema_version": 1,
        "status": "pass",
        "world_size": dist.get_world_size(),
        "batch_size_per_rank": 6,
        "global_batch_size": 6 * dist.get_world_size(),
        "learning_rate": 5.0e-4,
        "action_chunk": 25,
        "action_dimension": 7,
        "torch_version": torch.__version__,
        "openvla_oft_version": version("openvla-oft"),
        "cases": cases,
    }
    if dist.get_rank() == 0:
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        print(serialized, end="")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
