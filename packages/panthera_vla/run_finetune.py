#!/usr/bin/env python3
"""Register Panthera's RLDS contract, then run OpenVLA-OFT finetune.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import runpy

from checkpoint import (
    ObservationContract,
    PolicyContract,
    write_training_record,
)


# OpenVLA-OFT resizes every image to this before the vision encoder.
IMAGE_SIZE = (224, 224)

# Hyperparameters worth carrying with the weights: they decide what the
# checkpoint is, not merely how long it took to make.
OPTIMISATION_KEYS = (
    "batch_size", "learning_rate", "lora_rank", "max_steps", "grad_accumulation_steps",
    "val_freq", "early_stopping_min_delta", "early_stopping_patience", "image_aug",
    "save_freq", "save_latest_checkpoint_only",
)


def _options(argv) -> dict:
    """Collect draccus-style ``--key value`` arguments.

    A bare flag keeps the value ``"true"`` so it reads the same as an explicit
    ``--flag true``.
    """
    options: dict[str, str] = {}
    key = None
    for token in argv:
        if token.startswith("--"):
            key = token[2:]
            options[key] = "true"
        elif key is not None:
            options[key] = token
            key = None
    return options


def _flag(options: dict, name: str, default: bool) -> bool:
    raw = options.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _source_dataset(options: dict):
    """Locate the dataset the RLDS was built from, through the access layer."""
    root = os.environ.get("PANTHERA_SOURCE_DATASET_ROOT")
    if not root:
        raise SystemExit(
            "PANTHERA_SOURCE_DATASET_ROOT is unset, so the checkpoint cannot record "
            "which dataset produced it.  A checkpoint without that is not "
            "reproducible and evaluation cannot check it; set the variable to the "
            "dataset root that --data_root_dir was converted from."
        )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "panthera_sim"))
    from dataset import open_dataset

    return root, open_dataset(Path(root))


def _validate_action_contract(*, model_dim: int, model_chunk: int,
                              model_proprio: int, rlds_dim: int,
                              rlds_chunk: int, rlds_proprio: int, dataset) -> None:
    """Refuse silent action-chunk or proprioception-width mismatches."""
    if model_dim != rlds_dim:
        raise SystemExit(
            f"prismatic resolved ACTION_DIM={model_dim}, but the RLDS was "
            f"written with {rlds_dim} action dimensions"
        )
    if rlds_chunk != model_chunk:
        raise SystemExit(
            f"RLDS was registered with action chunk {rlds_chunk}, but prismatic "
            f"resolved NUM_ACTIONS_CHUNK={model_chunk}; export "
            "PANTHERA_ACTION_CHUNK to the model's chunk length before conversion "
            "and training"
        )
    if rlds_proprio != model_proprio:
        raise SystemExit(
            f"RLDS proprioception has {rlds_proprio} dimensions, but prismatic "
            f"resolved PROPRIO_DIM={model_proprio}; export PANTHERA_PROPRIO_DIM "
            "before conversion and training"
        )
    if dataset.contract.action_chunk != model_chunk:
        raise SystemExit(
            f"dataset {dataset.name} declares action chunk "
            f"{dataset.contract.action_chunk}, but the model resolved {model_chunk}"
        )


def _write_contract(options: dict, run_dir: Path, note):
    """Write the contract beside the weights, before and after training.

    Nothing wrote it before: ``checkpoint.py`` existed with no caller, so every
    run produced weights that could not say what action chunk, unnorm key or
    image preparation they were trained under, and the evaluation harness fell
    back to its own defaults.  That is how evaluation came to center-crop images
    for a model trained with ``--image_aug false``.
    """
    # Read what prismatic actually resolved, not what we meant: an unset
    # ROBOT_PLATFORM falls back to LIBERO silently, which swaps the action
    # chunk to 8 and the proprio width to 8 with nothing saying so.
    from prismatic.vla.constants import (
        ACTION_DIM,
        NUM_ACTIONS_CHUNK,
        PROPRIO_DIM,
        ROBOT_PLATFORM,
    )

    # Imported here rather than at module scope so the argument and contract
    # helpers stay importable without h5py or TensorFlow, and can be tested.
    from panthera_rlds import ACTION_CHUNK, ACTION_DIMENSION, PROPRIO_DIMENSION

    dataset_root, dataset = _source_dataset(options)
    resolved_chunk = int(NUM_ACTIONS_CHUNK)
    _validate_action_contract(
        model_dim=int(ACTION_DIM), model_chunk=resolved_chunk,
        model_proprio=int(PROPRIO_DIM), rlds_dim=ACTION_DIMENSION,
        rlds_chunk=ACTION_CHUNK, rlds_proprio=PROPRIO_DIMENSION,
        dataset=dataset,
    )
    # Training-time random crops are what a center crop at evaluation exists to
    # match, so the evaluation flag is the training flag.
    image_aug = _flag(options, "image_aug", False)
    policy = PolicyContract(
        action_dim=int(ACTION_DIM),
        action_chunk=int(NUM_ACTIONS_CHUNK),
        proprio_dim=int(PROPRIO_DIM),
        use_proprio=_flag(options, "use_proprio", True),
        use_l1_regression=not _flag(options, "use_diffusion", False),
        num_images_in_input=int(options.get("num_images_in_input", 1)),
        unnorm_key=options["dataset_name"],
        robot_platform=str(ROBOT_PLATFORM),
    )
    observation = ObservationContract(image_size=IMAGE_SIZE, center_crop=image_aug)
    run_dir.mkdir(parents=True, exist_ok=True)
    return write_training_record(
        run_dir,
        policy,
        observation,
        dataset_name=dataset.name,
        dataset_digest=dataset.digest(),
        dataset_root=dataset_root,
        base_model=options.get("vla_path", ""),
        optimisation={k: options[k] for k in OPTIMISATION_KEYS if k in options},
        splits={"rlds_root": options.get("data_root_dir", "")},
        dataset_lineage=dataset.manifest.get("subset", {}),
        note=note,
    )


def main() -> int:
    from panthera_rlds import register_openvla_dataset

    register_openvla_dataset()
    # Resolving the workspace by counting parent directories broke the moment
    # this module moved one level deeper during the 2026-09-19 restructure, and
    # it broke with a missing-file error rather than anything that named the
    # cause.  Walk up for the checkout instead, and let the environment override.
    override = os.environ.get("PANTHERA_OPENVLA_FINETUNE")
    if override:
        script = Path(override)
    else:
        relative = Path(
            "runtime/rlinf/.venv/lib/python3.11/site-packages/vla-scripts/finetune.py"
        )
        roots = []
        if os.environ.get("PANTHERA_VLA_ROOT"):
            roots.append(Path(os.environ["PANTHERA_VLA_ROOT"]))
        roots.extend(Path(__file__).resolve().parents)
        for root in roots:
            candidate = root / relative
            if candidate.is_file():
                script = candidate
                break
        else:
            raise FileNotFoundError(
                f"could not find {relative} above {Path(__file__).resolve().parent}; "
                "set PANTHERA_VLA_ROOT or PANTHERA_OPENVLA_FINETUNE"
            )
    if not script.is_file():
        raise FileNotFoundError(f"OpenVLA-OFT finetune.py not found: {script}")
    from experiments.robot import openvla_utils

    package_root = Path(openvla_utils.__file__).resolve().parents[2]
    if not (package_root / "prismatic").is_dir():
        raise FileNotFoundError(
            f"OpenVLA-OFT package root lacks prismatic/: {package_root}"
        )
    os.chdir(package_root)
    options = _options(sys.argv[1:])
    run_dir = Path(options["run_root_dir"]) / options.get("run_id_override", "run")
    # Written first so a run killed by its timeout still carries its contract,
    # then rewritten so the finished checkpoint also carries how it stopped.
    _write_contract(options, run_dir, note="written before training")
    runpy.run_path(str(script), run_name="__main__")
    _write_contract(options, run_dir, note=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
