#!/usr/bin/env bash
# Render one checkpoint in both deployment-relevant execution modes:
#   1. predict 25 actions, execute 20, then replan;
#   2. predict 25 actions every control step and equally average every
#      historical prediction that covers the current step.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
run_root="${CI_OVERFIT_RUN_ROOT:?set CI_OVERFIT_RUN_ROOT}"
step="${CI_OVERFIT_PREVIEW_STEP:?set CI_OVERFIT_PREVIEW_STEP}"
gpu="${CI_OVERFIT_EVAL_GPU:-0}"
min_free_mib="${CI_OVERFIT_EVAL_MIN_FREE_MIB:-28000}"
pair_root="${CI_OVERFIT_PREVIEW_RESULTS:-${run_root}/paired-preview-step-${step}}"
evaluator="${workspace}/pipelines/ci/evaluate_numbered_overfit_checkpoints.sh"
mosaic="${workspace}/packages/panthera_sim/mosaic_videos.py"
resume_unit="${CI_OVERFIT_RESUME_UNIT:-}"

resume_queue() {
  if [[ -n "$resume_unit" ]]; then
    systemctl --user restart "$resume_unit" || true
  fi
}
trap resume_queue EXIT

mkdir -p "$pair_root"

CI_OVERFIT_EVAL_GPU="$gpu" \
CI_OVERFIT_EVAL_MIN_FREE_MIB="$min_free_mib" \
CI_OVERFIT_EVAL_ONLY_STEP="$step" \
CI_OVERFIT_EVAL_RESULTS="${pair_root}/standard-h20" \
CI_OVERFIT_EVAL_EXECUTION_HORIZON=20 \
CI_OVERFIT_EVAL_TEMPORAL_ENSEMBLE= \
CI_OVERFIT_EVAL_VARIANT_LABEL="mode h20: query 25 / execute 20" \
CI_OVERFIT_EVAL_REFRESH_MOSAIC=0 \
CI_OVERFIT_EVAL_PLOT_TEMPORAL_LOSS=1 \
bash "$evaluator"

CI_OVERFIT_EVAL_GPU="$gpu" \
CI_OVERFIT_EVAL_MIN_FREE_MIB="$min_free_mib" \
CI_OVERFIT_EVAL_ONLY_STEP="$step" \
CI_OVERFIT_EVAL_RESULTS="${pair_root}/every-step-equal" \
CI_OVERFIT_EVAL_EXECUTION_HORIZON=1 \
CI_OVERFIT_EVAL_TEMPORAL_ENSEMBLE=0.0 \
CI_OVERFIT_EVAL_VARIANT_LABEL="mode h1: query 25 / equal sliding mean" \
CI_OVERFIT_EVAL_REFRESH_MOSAIC=0 \
CI_OVERFIT_EVAL_PLOT_TEMPORAL_LOSS=0 \
CI_OVERFIT_EVAL_VALIDATION_REFERENCE="${pair_root}/standard-h20/step-${step}.json" \
bash "$evaluator"

python3 "$mosaic" \
  --input \
    "${pair_root}/standard-h20/step-${step}.mp4" \
    "${pair_root}/every-step-equal/step-${step}.mp4" \
  --fps "${CI_OVERFIT_EVAL_VIDEO_FPS:-25}" \
  --output "${pair_root}/step-${step}-two-mode-comparison.mp4"

python3 - \
  "${pair_root}/standard-h20/step-${step}.json" \
  "${pair_root}/every-step-equal/step-${step}.json" \
  "${pair_root}/pair-summary.json" <<'PY'
import json
import sys
from pathlib import Path

standard_path, ensemble_path, output_path = map(Path, sys.argv[1:])

def compact(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    case = report["cases"][0]
    offline_validation = dict(case.get("offline_validation") or {})
    offline_validation.pop("per_frame", None)
    return {
        "result": str(path),
        "execution_horizon": report.get("execution_horizon"),
        "temporal_ensemble": report.get("temporal_ensemble"),
        "success": case.get("success"),
        "completion_class": case.get("completion_class"),
        "on_time_success": case.get("on_time_success"),
        "delayed_success": case.get("delayed_success"),
        "diagnostic_success": case.get("diagnostic_success"),
        "expert_action_budget": case.get("expert_action_budget"),
        "evaluation_action_budget": case.get("evaluation_action_budget"),
        "executed_actions": case.get("executed_actions"),
        "policy_queries": case.get("policy_queries"),
        "inference_timing": case.get("inference_timing"),
        "offline_validation": offline_validation,
    }

summary = {
    "measurement": (
        "frame-ready lower-bound policy latency; simulator rendering, physics, "
        "and video encoding excluded"
    ),
    "standard_h20": compact(standard_path),
    "every_step_equal": compact(ensemble_path),
}
output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(output_path)
PY

echo "双版本预览完成：${pair_root}/step-${step}-two-mode-comparison.mp4"
