#!/usr/bin/env python3
"""Merge a saved LoRA adapter into its base VLA outside the training process.

The training wrapper defers the 7B merge to a clean shutdown, so a run that ends
on a wall-clock timeout leaves the adapter, action head and proprio projector on
disk with no merged model.  This performs exactly the merge that
``merge_saved_best_checkpoint`` would have done, and records that it happened
out of band rather than at the end of a normal run.

The merge is CPU-only, so it does not contend with training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = (
    "lora_adapter/adapter_model.safetensors",
    "lora_adapter/adapter_config.json",
    "action_head--latest_checkpoint.pt",
    "proprio_projector--latest_checkpoint.pt",
    "dataset_statistics.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--allow-running-training",
        action="store_true",
        help="skip the guard that refuses to merge while training still writes",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing merged model"
    )
    return parser.parse_args()


def _training_is_running(run_dir: Path) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "run_finetune.py"], capture_output=True, text=True
    )
    run_root = str(run_dir.parent)
    return result.returncode == 0 and any(
        run_root in line for line in result.stdout.splitlines()
    )


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()

    if not args.allow_running_training and _training_is_running(run_dir):
        raise SystemExit(
            "refusing to merge while run_finetune.py is still running; the "
            "adapter may be mid-write. Wait for training to exit."
        )

    missing = [name for name in REQUIRED if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(f"run dir is missing required artifacts: {missing}")

    index = run_dir / "model.safetensors.index.json"
    if index.exists() and not args.force:
        raise SystemExit(
            f"a merged model already exists at {run_dir}; pass --force to redo it"
        )

    # Record what the adapter looked like going in, so the provenance file can be
    # checked against the artifacts later.
    adapter = run_dir / "lora_adapter" / "adapter_model.safetensors"
    early_stopping_path = run_dir / "early-stopping.json"
    early_stopping = (
        json.loads(early_stopping_path.read_text(encoding="utf-8"))
        if early_stopping_path.is_file()
        else None
    )

    os.environ.setdefault("HF_HOME", "/data/lyy/panthera-vla/cache/huggingface")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq

    print(f"[merge] base model: {args.base_model}")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        str(args.base_model),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"[merge] adapter: {run_dir / 'lora_adapter'}")
    merged = PeftModel.from_pretrained(base_vla, str(run_dir / "lora_adapter"))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(run_dir))
    print(f"[merge] merged model written to {run_dir}")

    shards = sorted(run_dir.glob("model-*.safetensors"))
    if not shards or not index.exists():
        raise SystemExit("merge produced no safetensors shards or index")

    provenance = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "merged_by": "merge_lora_checkpoint.py (out of band)",
        "reason": (
            "training stopped on a wall-clock budget, so the wrapper's "
            "end-of-run merge never ran"
        ),
        "run_dir": str(run_dir),
        "base_model": str(args.base_model),
        "adapter_bytes": adapter.stat().st_size,
        "shard_count": len(shards),
        "merged_bytes": sum(shard.stat().st_size for shard in shards),
        "early_stopping_at_merge": early_stopping,
    }
    (run_dir / "manual-merge.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
