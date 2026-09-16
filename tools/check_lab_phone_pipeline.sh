#!/usr/bin/env bash

set -euo pipefail

root="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
local_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

exec "${local_root}/tools/lab_ssh.sh" "
root='${root}'
printf 'pipeline_process='
ps -p 318232 -o pid=,stat=,etime= || true
for marker in \
  .panthera-phone-media-audit-state/media.ok \
  .panthera-phone-openvla-sft-state/train.ok \
  .panthera-phone-policy-eval-state/eval.completed \
  .panthera-phone-policy-eval-state/eval.ok \
  .panthera-phone-policy-pipeline-state/pipeline.ok; do
  printf '%s=' \"\${marker}\"
  if test -f \"\${root}/\${marker}\"; then echo yes; else echo no; fi
done
if test -s \"\${root}/.panthera-phone-openvla-sft-state/run-log.txt\"; then
  log=\$(cat \"\${root}/.panthera-phone-openvla-sft-state/run-log.txt\")
  printf 'train_progress='
  tr '\\r' '\\n' < \"\${log}\" | grep -o '[0-9][0-9]*/5000' | tail -1 || true
fi
if test -s \"\${root}/.panthera-phone-policy-eval-state/run-log.txt\"; then
  eval_log=\$(cat \"\${root}/.panthera-phone-policy-eval-state/run-log.txt\")
  eval_root=\$(sed -n 's|.*runner.logger.log_path=\\([^ ]*\\).*|\\1|p' \"\${eval_log}\" | head -1)
  printf 'eval_videos='
  if test -n \"\${eval_root}\" && test -d \"\${eval_root}/video/eval\"; then
    find \"\${eval_root}/video/eval\" -type f -name '*.mp4' -size +0c | wc -l
  else
    echo 0
  fi
fi
"
