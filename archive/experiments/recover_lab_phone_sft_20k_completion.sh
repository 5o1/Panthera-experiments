#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
state="${workspace}/.panthera-phone-openvla-sft-20k-state"
model="${workspace}/runs/panthera-phone-openvla-sft/panthera-phone-wide-v3-vertical-sft-20000steps"
python="${workspace}/RLinf/.venv/bin/python"
final_val_l1="0.048022073412698416"
best_val_l1="0.040761498158959096"
best_val_step="19000"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：恢复脚本必须使用普通用户运行。" >&2
  exit 1
fi
if [[ ! -x "$python" || ! -d "$state" || ! -d "$model" ]]; then
  echo "错误：20k 恢复所需的环境、状态目录或模型目录不存在。" >&2
  exit 1
fi

exec 9>"${state}/recovery.lock"
if ! flock -n 9; then
  echo "错误：另一个 20k 收尾恢复正在运行。" >&2
  exit 1
fi
if ps -u "$(id -u)" -o comm=,args= | awk \
  '$1 ~ /^(python|torchrun)/ && /run_finetune[.]py/ { found=1 } END { exit !found }'; then
  echo "错误：OpenVLA 训练仍在运行，拒绝恢复完成标记。" >&2
  exit 1
fi

log=$(<"${state}/run-log.txt")
grep -q "Max step 20000 reached" "$log"
if grep -Eq 'Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|nan|NaN' "$log"; then
  echo "错误：训练日志包含原产物门禁止的异常。" >&2
  exit 1
fi
for required in \
  "${model}/config.json" \
  "${model}/dataset_statistics.json" \
  "${model}/model.safetensors.index.json" \
  "${model}/action_head--latest_checkpoint.pt" \
  "${model}/proprio_projector--latest_checkpoint.pt"; do
  if [[ ! -s "$required" ]]; then
    echo "错误：最终模型缺少文件：${required}" >&2
    exit 1
  fi
done
for gpu in 1 2 3; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "错误：GPU ${gpu} 仍有计算进程，拒绝恢复完成标记。" >&2
    exit 1
  fi
done

"$python" - "$model" <<'PY'
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

model = Path(sys.argv[1])
stats = json.loads((model / "dataset_statistics.json").read_text())
if "panthera_phone_vertical_cylinder" not in stats:
    raise SystemExit("missing Panthera phone normalization statistics")
for component in ("action_head", "proprio_projector"):
    value = torch.load(
        model / f"{component}--latest_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not value:
        raise SystemExit(f"empty {component} checkpoint")
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
if not shards:
    raise SystemExit("empty safetensors shard index")
for shard in shards:
    path = model / shard
    if not path.is_file():
        raise SystemExit(f"missing shard: {shard}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not handle.keys():
            raise SystemExit(f"empty shard: {shard}")
PY

"$python" - "$state" "$model" "$log" "$final_val_l1" "$best_val_l1" "$best_val_step" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

state, model, log = map(Path, sys.argv[1:4])
final_val_l1, best_val_l1 = map(float, sys.argv[4:6])
best_val_step = int(sys.argv[6])
now = datetime.now(timezone.utc).astimezone().isoformat()
summary = {
    "status": "passed",
    "model": str(model),
    "initial_model": str(model.parent / "panthera-phone-wide-v3-vertical-sft-5000steps"),
    "initialization": "resume_from_step_5000",
    "resume_step": 5000,
    "configured_max_steps": 20000,
    "gpu_count": 3,
    "physical_gpus": [1, 2, 3],
    "per_gpu_batch_size": 1,
    "effective_batch_size": 3,
    "objective": "l1_regression",
    "action_dimension": 7,
    "action_chunk": 5,
    "use_proprio": True,
    "image_augmentation": True,
    "lora_rank": 32,
    "validation_enabled": True,
    "dataset": "panthera_phone_vertical_cylinder",
    "task_schema_version": 3,
    "scene_profile": "phone_srt_vertical_socket",
    "final_validation_l1": final_val_l1,
    "best_validation_l1": best_val_l1,
    "best_validation_step": best_val_step,
    "completion_recovered": True,
    "original_wrapper_exit_observed": False,
}
recovery = {
    "status": "recovered_after_wrapper_exit",
    "recovered_at": now,
    "reason": "outer shell disappeared after model save and before status artifacts",
    "original_wrapper_exit_observed": False,
    "evidence": {
        "max_step_log": True,
        "fatal_log_patterns": False,
        "model_components_deserialized": True,
        "safetensors_index_and_headers_valid": True,
        "training_gpu_compute_processes": [],
        "final_validation_l1": final_val_l1,
    },
    "log": str(log),
    "model": str(model),
}
(state / "train-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(state / "completion-recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
(state / "final-model.txt").write_text(str(model) + "\n")
(state / "completed-at.txt").write_text(now + "\n")
(state / "train.ok").touch()
print(json.dumps(recovery, indent=2))
PY

echo "20k 模型收尾已从完整证据恢复。"
