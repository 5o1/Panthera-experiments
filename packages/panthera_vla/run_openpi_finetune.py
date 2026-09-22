#!/usr/bin/env python3
"""Run Panthera π0.5 stats or training against a read-only OpenPI checkout.

Examples (on the Lab, after converting the data):

  python run_openpi_finetune.py stats --openpi-root /data/lyy/openpi \
    --lerobot-home /data/lyy/panthera-vla/lerobot \
    --repo-id panthera/schema10 \
    --checkpoint-base-dir /data/lyy/panthera-vla/openpi-checkpoints \
    --assets-base-dir /data/lyy/panthera-vla/openpi-assets

  python run_openpi_finetune.py train --openpi-root /data/lyy/openpi \
    --lerobot-home /data/lyy/panthera-vla/lerobot \
    --repo-id panthera/schema10 --exp-name baseline --lora \
    --checkpoint-base-dir /data/lyy/panthera-vla/openpi-checkpoints \
    --assets-base-dir /data/lyy/panthera-vla/openpi-assets

The wrapper imports and calls upstream functions; it never writes into the
OpenPI checkout.  Checkpoints, assets and LeRobot data stay under explicit
``/data/lyy`` paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from openpi_config import build_train_config, register_config


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import OpenPI script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stats", "train"))
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--lerobot-home", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--exp-name", default="baseline")
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=("jax", "pytorch"), default="jax")
    parser.add_argument("--pytorch-weight-path")
    parser.add_argument("--action-horizon", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-train-steps", type=int, default=30_000)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--keep-period", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-stats-frames", type=int)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    root = cli.openpi_root.resolve()
    if not (root / "src/openpi").is_dir():
        raise SystemExit(f"not an OpenPI checkout: {root}")
    os.environ["HF_LEROBOT_HOME"] = str(cli.lerobot_home.resolve())
    os.environ["LEROBOT_HOME"] = str(cli.lerobot_home.resolve())
    sys.path.insert(0, str(root / "src"))

    conversion_path = cli.lerobot_home.resolve() / cli.repo_id / "panthera_conversion.json"
    if not conversion_path.is_file():
        raise SystemExit(f"missing Panthera LeRobot provenance: {conversion_path}")
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    if conversion.get("repo_id") != cli.repo_id:
        raise SystemExit(
            f"conversion repo_id is {conversion.get('repo_id')!r}, expected {cli.repo_id!r}"
        )
    if int(conversion.get("action_dim", -1)) != 7:
        raise SystemExit("conversion does not use the 7-D Panthera contract")
    if int(conversion.get("action_horizon", -1)) != cli.action_horizon:
        raise SystemExit(
            f"conversion action horizon is {conversion.get('action_horizon')}, "
            f"but --action-horizon is {cli.action_horizon}"
        )

    config = build_train_config(
        repo_id=cli.repo_id,
        exp_name=cli.exp_name,
        checkpoint_base_dir=cli.checkpoint_base_dir,
        assets_base_dir=cli.assets_base_dir,
        action_horizon=cli.action_horizon,
        batch_size=cli.batch_size,
        num_train_steps=cli.num_train_steps,
        save_interval=cli.save_interval,
        keep_period=cli.keep_period,
        num_workers=cli.num_workers,
        fsdp_devices=cli.fsdp_devices,
        lora=cli.lora,
        resume=cli.resume,
        overwrite=cli.overwrite,
        wandb_enabled=not cli.no_wandb,
        pytorch_weight_path=cli.pytorch_weight_path,
        source_dataset_digest=conversion["source_digest"],
    )
    register_config(config)

    run_contract = {
        "schema_version": 1,
        "openpi_config": config.name,
        "experiment": cli.exp_name,
        "repo_id": cli.repo_id,
        "source_dataset_digest": conversion["source_digest"],
        "action_dim": 7,
        "action_horizon": cli.action_horizon,
        "external_action_semantics": "absolute_joint_position_plus_absolute_gripper",
        "lora": bool(cli.lora),
        "engine": cli.engine,
        "base_checkpoint": "gs://openpi-assets/checkpoints/pi05_base/params",
    }
    contract_path = (
        cli.checkpoint_base_dir.resolve()
        / config.name
        / f"{cli.exp_name}.panthera-training.json"
    )
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(run_contract, indent=2) + "\n"
    if contract_path.exists() and contract_path.read_text(encoding="utf-8") != serialized:
        raise SystemExit(f"refusing to change an existing training contract: {contract_path}")
    contract_path.write_text(serialized, encoding="utf-8")

    if cli.stage == "stats":
        script = _load_script(root / "scripts/compute_norm_stats.py", "openpi_compute_norm_stats")
        script.main(config.name, max_frames=cli.max_stats_frames)
        return 0

    if cli.engine == "jax":
        script = _load_script(root / "scripts/train.py", "openpi_train")
        script.main(config)
    else:
        if not cli.pytorch_weight_path:
            raise SystemExit("--pytorch-weight-path is required for PyTorch training")
        script = _load_script(root / "scripts/train_pytorch.py", "openpi_train_pytorch")
        script.train_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
