#!/usr/bin/env bash
# round 0 状态速览。只读，不改任何东西。
W=/data/lyy/panthera-vla
S="$W/.panthera-v2-expanded-fixed-openvla-earlystop-state"
R="$W/runs/panthera-v2-expanded-fixed-openvla/panthera-v2-expanded-fixed-25x7-sft-earlystop-md1e-3-p3"
echo "==================== round 0 状态 ===================="
echo "现在: $(date --iso-8601=seconds)"
echo
echo "-- 链路阶段 --"
for m in ".panthera-v2-single-grasp-expanded_fixed-state/dataset.ok:数据集人工批准" \
         ".panthera-v2-expanded-fixed-media-audit-state/media.ok:媒体审计" \
         ".panthera-v2-expanded-fixed-rlds-state/rlds.ok:RLDS 4.1.0" \
         ".panthera-v2-expanded-fixed-openvla-smoke-state/train.ok:一步优化器 smoke"; do
  p="${m%%:*}"; n="${m##*:}"
  printf "  %-22s %s\n" "$n" "$([ -f "$W/$p" ] && echo 通过 || echo 未完成)"
done
echo
echo "-- 训练进程 --"
if pgrep -f "[f]inetune.py" >/dev/null; then
  echo "  运行中，已持续 $(ps -eo etime,args | grep '[f]inetune.py' | head -1 | awk '{print $1}')"
else
  echo "  未在运行"
fi
echo "  日志末尾:"
grep -E "TRAIN_EXIT|ELAPSED_MIN|ROUND0_DONE|ROUND0_ABORTED|SMOKE_FAILED" "$W/tmp/round0.log" 2>/dev/null | tail -4 | sed 's/^/    /'
echo
echo "-- checkpoint 阶梯 --"
if [ -d "$R" ]; then
  ls -d "$R"/*steps* 2>/dev/null | sed 's|.*/|    |' | tail -20
  echo "    (共 $(ls -d "$R"/*steps* 2>/dev/null | wc -l) 个)"
else
  echo "    尚未产生"
fi
echo
echo "-- 验证 loss 轨迹（仅作发散报警，不用于选模）--"
grep -oE "step=[0-9]+, loss=[0-9.]+" "$S"/train-*.log 2>/dev/null | tail -8 | sed 's/^/    /'
echo
echo "-- GPU --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    /'
echo "====================================================="
