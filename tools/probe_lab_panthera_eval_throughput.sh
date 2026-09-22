#!/usr/bin/env bash
# Sample GPU and CPU while a deliberately tiny closed-loop eval runs.
#
# Measuring utilisation does not need a scoring run: the duty cycle is set by
# one policy forward against one chunk of physics, and that ratio is visible
# within seconds.  A full 18-trajectory eval at the real action budget takes
# ten minutes or more and answers the same question no better, so this probe
# runs the shortest configuration that still exercises the real loop and marks
# its own result as non-scoring.
set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
trajectories="${PROBE_TRAJECTORIES:-3}"
budget="${PROBE_BUDGET:-200}"
samples="${PROBE_SAMPLES:-40}"
interval="${PROBE_INTERVAL:-2}"

export PANTHERA_EVAL_TRAJECTORIES="$trajectories"
export PANTHERA_EVAL_MAX_EPISODE_STEPS="$budget"
export PANTHERA_EVAL_ALLOW_SHORT_BUDGET=1
export PANTHERA_EVAL_MIN_SUCCESS=0
export PANTHERA_EVAL_STATE_ROOT="${workspace}/state/panthera-eval-throughput-probe-state"
export PANTHERA_EVAL_LOG_ROOT="${workspace}/logs/panthera-eval-throughput-probe"
rm -rf "$PANTHERA_EVAL_STATE_ROOT"

sample_file="$(mktemp)"
trap 'rm -f "$sample_file"' EXIT
(
  for _ in $(seq 1 "$samples"); do
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader \
      | awk -F", " '{printf "%s ", $2}' >>"$sample_file"
    printf '| %s\n' "$(awk '{print $1}' /proc/loadavg)" >>"$sample_file"
    sleep "$interval"
  done
) &
sampler=$!

start=$(date +%s)
bash "${workspace}/bin/run_lab_panthera_policy_eval.sh" >/dev/null 2>&1 || true
elapsed=$(( $(date +%s) - start ))
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true

echo "轨迹=${trajectories} 预算=${budget} 墙钟=${elapsed}s"
python3 - "$sample_file" <<'PY'
import statistics
import sys

rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    gpu_part, _, load_part = line.partition("|")
    values = [int(v.rstrip("%")) for v in gpu_part.split() if v.rstrip("%").isdigit()]
    if values:
        rows.append((values, float(load_part or 0)))
if not rows:
    raise SystemExit("采样为空")
width = len(rows[0][0])
for gpu in range(width):
    series = [r[0][gpu] for r in rows if len(r[0]) > gpu]
    idle = sum(1 for v in series if v == 0) / len(series)
    print(f"  GPU{gpu} 均值={statistics.mean(series):5.1f}%  "
          f"峰值={max(series):3d}%  零占用比例={idle:.0%}")
print(f"  负载均值={statistics.mean(r[1] for r in rows):.1f}  采样数={len(rows)}")
PY
