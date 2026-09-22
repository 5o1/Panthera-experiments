#!/usr/bin/env python3
"""Build an OpenPI π0.5 configuration without modifying the upstream checkout.

OpenPI's stock CLI only exposes configs registered in its source tree.  Panthera
keeps upstream repositories read-only, so :func:`build_train_config` constructs
the equivalent config at runtime and the launcher registers it only in memory.
All OpenPI imports are lazy: importing Panthera itself never requires JAX.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from openpi_adapter import ACTION_DIM, DEFAULT_ACTION_HORIZON, PantheraInputs, PantheraOutputs


BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base/params"


def build_train_config(
    *,
    repo_id: str,
    exp_name: str,
    checkpoint_base_dir: Path,
    assets_base_dir: Path,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    batch_size: int = 32,
    num_train_steps: int = 30_000,
    save_interval: int = 5_000,
    keep_period: int = 5_000,
    num_workers: int = 8,
    fsdp_devices: int = 1,
    lora: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    wandb_enabled: bool = True,
    pytorch_weight_path: str | None = None,
    source_dataset_digest: str | None = None,
):
    """Return a stock OpenPI ``TrainConfig`` for Panthera's 7-D contract."""
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must look like 'owner/dataset'")
    if not exp_name:
        raise ValueError("exp_name must be nonempty")
    for name, value in {
        "action_horizon": action_horizon,
        "batch_size": batch_size,
        "num_train_steps": num_train_steps,
        "save_interval": save_interval,
        "keep_period": keep_period,
        "num_workers": num_workers,
        "fsdp_devices": fsdp_devices,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")

    from typing_extensions import override
    from openpi import transforms as openpi_transforms
    from openpi.models import pi0_config
    from openpi.training import config as config_module
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    @dataclasses.dataclass(frozen=True)
    class PantheraLeRobotDataConfig(config_module.DataConfigFactory):
        """Repack LeRobot fields and preserve Panthera's external semantics."""

        @override
        def create(self, assets_dirs, model_config):
            repack = openpi_transforms.Group(
                inputs=[
                    openpi_transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            transforms = openpi_transforms.Group(
                inputs=[PantheraInputs()], outputs=[PantheraOutputs()]
            )
            # Recorded actions are absolute joint targets.  π models learn joint
            # deltas more reliably; the output transform restores absolute
            # targets before the policy reaches Panthera's executor.  Gripper
            # dimension 7 remains absolute in both directions.
            joint_mask = openpi_transforms.make_bool_mask(6, -1)
            transforms = transforms.push(
                inputs=[openpi_transforms.DeltaActions(joint_mask)],
                outputs=[openpi_transforms.AbsoluteActions(joint_mask)],
            )
            model_transforms = config_module.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=transforms,
                model_transforms=model_transforms,
                action_sequence_keys=("actions",),
            )

    model_kwargs = {
        "pi05": True,
        "action_dim": 32,
        "action_horizon": int(action_horizon),
    }
    if lora:
        model_kwargs.update(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
    model = pi0_config.Pi0Config(**model_kwargs)

    train_kwargs = dict(
        name="pi05_panthera",
        exp_name=exp_name,
        model=model,
        data=PantheraLeRobotDataConfig(
            repo_id=repo_id,
            base_config=config_module.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(BASE_CHECKPOINT),
        pytorch_weight_path=pytorch_weight_path,
        ema_decay=None if lora else 0.999,
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=min(1_000, max(1, num_train_steps // 10)),
            peak_lr=5e-5,
            decay_steps=max(num_train_steps, 1_000),
            decay_lr=5e-6,
        ),
        checkpoint_base_dir=str(Path(checkpoint_base_dir).resolve()),
        assets_base_dir=str(Path(assets_base_dir).resolve()),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        num_train_steps=int(num_train_steps),
        save_interval=int(save_interval),
        keep_period=int(keep_period),
        fsdp_devices=int(fsdp_devices),
        resume=bool(resume),
        overwrite=bool(overwrite),
        wandb_enabled=bool(wandb_enabled),
        policy_metadata={
            "robot": "panthera",
            "external_action_dim": ACTION_DIM,
            "action_semantics": "absolute_joint_position_plus_absolute_gripper",
            "camera_count": 1,
            "source_dataset_digest": source_dataset_digest,
        },
    )
    if lora:
        train_kwargs["freeze_filter"] = model.get_freeze_filter()
    return config_module.TrainConfig(**train_kwargs)


def register_config(config) -> None:
    """Register one generated config in the current OpenPI process only."""
    from openpi.training import config as config_module

    existing = config_module._CONFIGS_DICT.get(config.name)
    if existing is not None and existing is not config:
        raise ValueError(f"OpenPI config name already registered: {config.name}")
    config_module._CONFIGS_DICT[config.name] = config
