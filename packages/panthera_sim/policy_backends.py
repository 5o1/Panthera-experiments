#!/usr/bin/env python3
"""Policy loaders sharing Panthera's ``image,state,prompt -> [N,7]`` API."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class PolicyBackend(Protocol):
    name: str

    def predict(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        """Return an absolute Panthera action chunk with shape ``[N, 7]``."""


class OpenVLAOFTBackend:
    name = "openvla-oft"

    def __init__(self, config: dict):
        import torch
        from omegaconf import OmegaConf
        from rlinf.models.embodiment.openvla_oft.official import get_model

        self.proprio_dim = int(config["proprio_dim"])
        cfg = OmegaConf.create(
            {
                "model_path": config["model"],
                "action_dim": 7,
                "num_action_chunks": config["action_chunk"],
                "add_value_head": False,
                "value_type": "action_level",
                "proprio_dim": self.proprio_dim,
                "use_proprio": config["use_proprio"],
                "use_film": False,
                "use_l1_regression": config["use_l1_regression"],
                "num_images_in_input": config["num_images_in_input"],
                "max_prompt_length": 512,
                "unnorm_key": config["unnorm_key"],
            }
        )
        self.native_model = get_model(cfg, torch_dtype=torch.bfloat16).to("cuda").eval()

    def predict(self, image, state, instruction) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.proprio_dim,):
            raise ValueError(
                f"OpenVLA checkpoint expects {self.proprio_dim}-D proprioception, "
                f"got {state.shape}"
            )
        predicted, _ = self.native_model.predict_action_batch(
            env_obs={
                "main_images": [image],
                "wrist_images": None,
                "states": state[None, :],
                "task_descriptions": [instruction],
            },
            do_sample=False,
            temperature=-1.0,
            top_k=-1,
            calulate_logprobs=False,
            calulate_values=False,
        )
        return predicted.float().cpu().numpy()[0]


class OpenPI05Backend:
    name = "openpi-pi05"

    def __init__(self, config: dict):
        from openpi.policies import policy_config
        from openpi_adapter import inference_observation
        from openpi_config import build_train_config

        model_path = Path(config["model"]).resolve()
        train_config = build_train_config(
            repo_id=config["openpi_repo_id"],
            exp_name="inference",
            checkpoint_base_dir=model_path.parent,
            assets_base_dir=model_path.parent,
            action_horizon=config["action_chunk"],
            batch_size=1,
            num_train_steps=1,
            save_interval=1,
            keep_period=1,
            num_workers=1,
            fsdp_devices=1,
            lora=config["openpi_lora"],
            wandb_enabled=False,
            source_dataset_digest=config.get("dataset_digest"),
        )
        self._observation = inference_observation
        self.native_model = policy_config.create_trained_policy(
            train_config,
            model_path,
            pytorch_device=config.get("openpi_device"),
        )

    def predict(self, image, state, instruction) -> np.ndarray:
        result = self.native_model.infer(self._observation(image, state, instruction))
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"π0.5 backend must return [N, 7], got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("π0.5 backend returned NaN or infinity")
        return actions


def create_backend(config: dict) -> PolicyBackend:
    """Load exactly one backend after the worker's GPU has been assigned."""
    name = config["policy_backend"]
    if name == OpenVLAOFTBackend.name:
        return OpenVLAOFTBackend(config)
    if name == OpenPI05Backend.name:
        return OpenPI05Backend(config)
    raise ValueError(f"unknown policy backend: {name}")
