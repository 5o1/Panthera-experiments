#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tools/lib/process_cleanup.sh
source "${PROJECT_ROOT}/tools/lib/process_cleanup.sh"

setsid bash -c 'trap "exit 0" TERM; while :; do sleep 0.1; done' &
test_pid=$!
sleep 0.1
kill -0 -- "-$test_pid"
stop_process_group "$test_pid" TERM
if kill -0 -- "-$test_pid" 2>/dev/null; then
    printf 'FAIL：进程组清理后仍存活。\n' >&2
    exit 1
fi

# Cleanup must be idempotent because EXIT and a user signal can race.
stop_process_group "$test_pid" TERM
printf 'PASS：进程组清理可终止子进程且重复调用安全。\n'
