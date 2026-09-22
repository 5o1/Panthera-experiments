#!/usr/bin/env bash
# 训练结束后补做 LoRA 合并。可反复运行：已合并会拒绝覆盖。
set -uo pipefail
W=/data/lyy/panthera-vla
R="$W/runs/panthera-v2-expanded-fixed-openvla/panthera-v2-expanded-fixed-25x7-sft-earlystop-md1e-3-p3"
BASE="$W/models/openvla-oft-place-empty-cup"
LOG="$W/tmp/merge-$(date +%Y%m%d-%H%M%S).log"

if pgrep -f run_finetune.py >/dev/null; then
  echo "训练仍在运行（$(ps -eo etime,args | grep '[t]imeout --signal' | head -1 | awk '{print $1}')），等待其退出..."
  while pgrep -f run_finetune.py >/dev/null; do sleep 30; done
  sleep 20
fi

echo "== 训练已结束，检查产物 =="
for f in lora_adapter/adapter_model.safetensors action_head--latest_checkpoint.pt \
         proprio_projector--latest_checkpoint.pt dataset_statistics.json early-stopping.json; do
  printf "  %-46s %s\n" "$f" "$([ -e "$R/$f" ] && echo OK || echo 缺失)"
done
echo "  最佳步/loss: $(python3 -c "import json;d=json.load(open('$R/early-stopping.json'));print(d['best_step'], d['best_loss'])" 2>/dev/null)"
echo
echo "== 开始合并（CPU，不占 GPU），日志 $LOG =="
source "$W/activate_lab_vla.sh" >/dev/null 2>&1
python "$W/panthera-openvla-adapter/merge_lora_checkpoint.py" \
  --run-dir "$R" --base-model "$BASE" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo "MERGE_EXIT=$STATUS"
if [ "$STATUS" -eq 0 ]; then
  echo "== 合并后目录 =="
  ls -la "$R" | grep -E "model-0|index.json|manual-merge" | sed 's/^/  /'
fi
exit $STATUS
